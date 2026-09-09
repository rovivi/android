// Orquestación de los campos que viven en C++. Sin JNI a propósito: el .so y
// el CLI de host (tools/host/cli.cpp) llaman exactamente a esto, así que lo
// que mide el test de paridad es lo que corre el teléfono.
#include "piu_ocr.h"
#include <algorithm>
#include <sstream>
#include <string>

namespace piu {
namespace {

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
    else { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; }
  }
  return o;
}

}  // namespace

const char* Engine::emptyJson() {
  return "{\"titles\":[],\"score\":-1,\"score_margin\":0,\"score_digits\":\"\","
         "\"chart_type\":\"\",\"chart_conf\":0,"
         "\"level_digits\":[],\"level_scores\":[]}";
}

bool Engine::load(const std::string& base) {
  chars_ = Templates::load(base + "/chars.bin");
  level_ = Templates::load(base + "/level.bin");
  digits_ = Templates::load(base + "/digits.bin");
  return !chars_.empty() && !level_.empty() && !digits_.empty() &&
         det_.load(base + "/piu_yolo.param", base + "/piu_yolo.bin");
}

std::string Engine::read(const cv::Mat& img, const std::vector<Box>* given,
                         std::vector<Box>* usedBoxes) const {
  // imgsz 1280 con TTA: es la única configuración que da cajas usables. Bajar a
  // 768 detecta MÁS cajas pero peor puestas, y el OCR consume el recorte —
  // medido, cuesta canción 0.800 -> 0.633.
  const std::vector<Box> boxes = given ? *given : det_.detect(img, 1280, augs);
  if (usedBoxes) *usedBoxes = boxes;

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
  std::stable_sort(titles.begin(), titles.end(),
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
    chars_.predict(gs, &lab, &mar);
    std::string txt;
    for (int L : lab) txt += char(L);
    if (!first) js << ",";
    js << "{\"raw\":\"" << esc(txt) << "\",\"conf\":" << b.conf << "}";
    first = false;
  }

  // difficulty: SOLO la bolita de mayor confianza, como _first() en
  // pipeline.py. Tomar el mejor voto entre todas las cajas parecía más
  // robusto y era peor (paridad: 0.881 -> 0.810): una segunda caja floja
  // sobre pantalla azul vota "halfdouble" con conf 1.0 y le gana a la real.
  const Box* diff = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 0 && (!diff || b.conf > diff->conf)) diff = &b;
  if (diff) {
    // pad +0.02: medido sobre 55 bolitas, -0.02 da 0.855 y +0.02 da 0.873.
    std::string t; float cf = 0.f;
    if (classifyChartType(cropBox(img, *diff, 0.02f), &t, &cf)) {
      chartType = t; chartConf = cf;
    }
    auto gs = segmentBadge(cropBox(img, *diff, -0.02f), 2);
    if (!gs.empty()) {
      std::vector<float> mar;
      level_.predict(gs, &lvlDigits, &mar);
      level_.scores(gs, &lvlScores);
      for (int& d : lvlDigits) d = digitOf(d);   // code point -> dígito
    }
  }

  // score: la caja de mayor confianza, como _first() en pipeline.py. Devuelve
  // el margen MÍNIMO entre los dígitos; el gate vive en Kotlin.
  const Box* scoreBox = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 3 && (!scoreBox || b.conf > scoreBox->conf)) scoreBox = &b;
  int scoreVal = -1;
  float scoreMargin = 0.f;
  std::string scoreDigits;
  if (scoreBox)
    readScore(cropBox(img, *scoreBox, 0.06f), digits_, &scoreVal, &scoreMargin,
              &scoreDigits);

  js << "],\"score\":" << scoreVal << ",\"score_margin\":" << scoreMargin
     << ",\"score_digits\":\"" << esc(scoreDigits) << "\"";
  js << ",\"chart_type\":\"" << esc(chartType) << "\",\"chart_conf\":"
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
  return js.str();
}

}  // namespace piu
