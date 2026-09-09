// Puente JNI. Deliberadamente delgado: acá solo se convierte el Bitmap a Mat,
// se corren los campos que viven en C++ y se devuelve JSON. Los gates y el
// matching contra el catálogo viven en Kotlin, porque son lógica de producto
// que conviene poder tocar sin recompilar el .so.
#include "piu_ocr.h"
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <exception>
#include <sstream>
#include <string>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "piuocr", __VA_ARGS__)

using namespace piu;

namespace {

struct Ctx {
  Templates chars, level;
  Detector det;
};

// Desbloquea los píxeles también si cvtColor tira: un lockPixels sin su
// unlock deja el Bitmap inutilizable para el resto del proceso.
struct PixelLock {
  JNIEnv* env; jobject bmp; void* px = nullptr;
  PixelLock(JNIEnv* e, jobject b) : env(e), bmp(b) {
    if (AndroidBitmap_lockPixels(env, bmp, &px) < 0) px = nullptr;
  }
  ~PixelLock() { if (px) AndroidBitmap_unlockPixels(env, bmp); }
};

bool bitmapToMat(JNIEnv* env, jobject bmp, cv::Mat* out) {
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bmp, &info) < 0) return false;
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("bitmap format %d no soportado (se espera ARGB_8888)", info.format);
    return false;
  }
  PixelLock lock(env, bmp);
  if (!lock.px) return false;
  // stride puede ser mayor que width*4: respetarlo.
  cv::Mat rgba(int(info.height), int(info.width), CV_8UC4, lock.px,
               size_t(info.stride));
  cv::cvtColor(rgba, *out, cv::COLOR_RGBA2BGR);
  return true;
}

// JSON válido siempre: además de " y \, escapa controles y todo lo que no sea
// ASCII como \uXXXX. Las etiquetas de chars.bin son códigos de carácter y un
// byte suelto >= 0x80 rompería el parser de JSONObject.
std::string esc(const std::string& s) {
  static const char* hex = "0123456789abcdef";
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c >= 0x20 && c < 0x7f) o += char(c);
    else {
      o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15];
    }
  }
  return o;
}

const char* kEmpty = "{\"titles\":[],\"chart_type\":\"\",\"chart_conf\":0,"
                     "\"level_digits\":[],\"level_scores\":[]}";

}  // namespace

extern "C" {

// Devuelve 0 si algún asset no cargó: un handle a medio inicializar solo
// produce lecturas vacías en silencio, que es lo peor que puede pasar.
JNIEXPORT jlong JNICALL
Java_com_piu_ocr_PiuOcr_nativeCreate(JNIEnv* env, jclass, jstring dir) {
  const char* d = env->GetStringUTFChars(dir, nullptr);
  if (!d) return 0;
  const std::string base(d);
  env->ReleaseStringUTFChars(dir, d);

  auto* c = new Ctx();
  try {
    c->chars = Templates::load(base + "/chars.bin");
    c->level = Templates::load(base + "/level.bin");
    const bool ok = !c->chars.empty() && !c->level.empty() &&
                    c->det.load(base + "/piu_yolo.param", base + "/piu_yolo.bin");
    if (!ok) {
      LOGE("nativeCreate: fallo cargando assets en %s (chars=%d level=%d)",
           base.c_str(), int(!c->chars.empty()), int(!c->level.empty()));
      delete c;
      return 0;
    }
  } catch (const std::exception& e) {
    LOGE("nativeCreate: %s", e.what());
    delete c;
    return 0;
  }
  return reinterpret_cast<jlong>(c);
}

JNIEXPORT void JNICALL
Java_com_piu_ocr_PiuOcr_nativeDestroy(JNIEnv*, jclass, jlong h) {
  delete reinterpret_cast<Ctx*>(h);
}

/**
 * Corre el detector y el OCR sobre el bitmap y devuelve JSON. Los gates y el
 * matching contra el catálogo viven en Kotlin: son lógica de producto que
 * conviene poder tocar sin recompilar el .so.
 */
JNIEXPORT jstring JNICALL
Java_com_piu_ocr_PiuOcr_nativeRead(JNIEnv* env, jclass, jlong h, jobject bmp) {
  auto* c = reinterpret_cast<Ctx*>(h);
  cv::Mat img;
  if (!c || !bitmapToMat(env, bmp, &img)) return env->NewStringUTF(kEmpty);

  // Una cv::Exception que cruza la frontera JNI aborta el proceso entero.
  try {
  // imgsz 1280 con TTA: es la única configuración que da cajas usables. Bajar a
  // 768 detecta MÁS cajas pero peor puestas, y el OCR consume el recorte —
  // medido, cuesta canción 0.800 -> 0.633.
  const std::vector<Box> boxes = c->det.detect(img, 1280, /*tta=*/true);

  std::ostringstream js;
  js.imbue(std::locale::classic());     // "0.5", nunca "0,5"
  js << "{\"titles\":[";
  std::string chartType;
  float chartConf = 0.f;
  std::vector<int> lvlDigits;
  std::vector<std::vector<float>> lvlScores;

  // song_name: hasta 3 cajas, de mayor a menor confianza.
  std::vector<Box> titles;
  for (const Box& b : boxes) if (b.cls == 4) titles.push_back(b);
  std::sort(titles.begin(), titles.end(),
            [](const Box& a, const Box& b) { return a.conf > b.conf; });
  if (titles.size() > 3) titles.resize(3);
  bool first = true;
  for (const Box& b : titles) {
    cv::Mat roi = cropBox(img, b, 0.04f);
    if (roi.empty()) continue;
    roi = focusBand(roi);
    auto gs = segmentChars(roi);
    if (gs.empty()) continue;
    std::vector<int> lab; std::vector<float> mar;
    c->chars.predict(gs, &lab, &mar);
    std::string txt;
    for (int L : lab) txt += char(L);
    if (!first) js << ",";
    js << "{\"raw\":\"" << esc(txt) << "\",\"conf\":" << b.conf << "}";
    first = false;
  }

  for (const Box& b : boxes) {
    if (b.cls != 0) continue;                          // difficulty
    // pad +0.02: medido sobre 55 bolitas, -0.02 da 0.855 y +0.02 da 0.873.
    std::string t; float cf = 0.f;
    if (classifyChartType(cropBox(img, b, 0.02f), &t, &cf) && cf > chartConf) {
      chartType = t; chartConf = cf;
    }
    if (lvlDigits.empty()) {
      auto gs = segmentBadge(cropBox(img, b, -0.02f), 2);
      if (!gs.empty()) {
        std::vector<float> mar;
        c->level.predict(gs, &lvlDigits, &mar);
        c->level.scores(gs, &lvlScores);
      }
    }
  }

  js << "],\"chart_type\":\"" << esc(chartType) << "\",\"chart_conf\":"
     << chartConf << ",\"level_digits\":[";
  for (size_t i = 0; i < lvlDigits.size(); ++i) js << (i ? "," : "") << lvlDigits[i];
  js << "],\"level_scores\":[";
  for (size_t i = 0; i < lvlScores.size(); ++i) {
    js << (i ? ",[" : "[");
    for (size_t j = 0; j < lvlScores[i].size(); ++j)
      js << (j ? "," : "") << lvlScores[i][j];
    js << "]";
  }
  js << "]}";
  return env->NewStringUTF(js.str().c_str());
  } catch (const std::exception& e) {
    LOGE("nativeRead: %s", e.what());
    return env->NewStringUTF(kEmpty);
  }
}

}  // extern "C"
