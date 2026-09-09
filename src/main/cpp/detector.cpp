// Detector YOLO sobre NCNN, con TTA implementado a mano.
//
// POR QUÉ EL TTA. Medido sobre 58 fotos de cabina, detecciones de song_name:
//   PyTorch con TTA     43 de 58
//   PyTorch sin TTA     29 de 58
//   NCNN    sin TTA     27 de 58
// El export a NCNN NO pierde nada (27 vs 29 es ruido); lo que se pierde es el
// TTA, que ultralytics aplica en PyTorch y NCNN ignora en silencio. Sin él, un
// tercio de los títulos no se detecta.
//
// Cuesta 3 pasadas. El detector ya es el 93 % del tiempo, así que esto es la
// decisión de latencia más cara del módulo — está tomada a conciencia.
#include "piu_ocr.h"
#include <cpu.h>
#include <net.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace piu {
namespace {

// Escalas del TTA. Las mismas que usa ultralytics para YOLO: la original más
// una reducción y un flip horizontal. Se pasó de 3 a 2 escalas porque la
// tercera no agregaba detecciones y sí un 33 % de costo.
struct Aug { float scale; bool flip; };
const Aug kAugs[] = {{1.00f, false}, {0.83f, false}, {1.00f, true}};

float iou(const Box& a, const Box& b) {
  const int x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
  const int x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
  const int w = std::max(0, x2 - x1), h = std::max(0, y2 - y1);
  const float inter = float(w) * h;
  const float ua = float(a.x2 - a.x1) * (a.y2 - a.y1)
                 + float(b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

}  // namespace

Detector::~Detector() { delete net_; }

bool Detector::load(const std::string& param, const std::string& bin) {
  delete net_;
  net_ = new ncnn::Net();
  net_->opt.use_vulkan_compute = false;   // arm64 CPU: predecible y sin drivers
  // Solo los cores grandes: un 4 fijo en un big.LITTLE 2+6 mete dos hilos en
  // cores chicos y el paso lo marca el más lento.
  ncnn::set_cpu_powersave(2);
  net_->opt.num_threads = std::max(1, std::min(4, ncnn::get_big_cpu_count()));
  net_->opt.lightmode = true;
  if (net_->load_param(param.c_str()) != 0 || net_->load_model(bin.c_str()) != 0) {
    delete net_;
    net_ = nullptr;
    return false;
  }
  return true;
}

std::vector<Box> Detector::detectOnce(const cv::Mat& bgr, int imgsz,
                                      float scale, bool flip) const {
  // Letterbox al tamaño del modelo, igual que ultralytics.
  const int W = bgr.cols, H = bgr.rows;
  const int target = int(imgsz * scale) / 32 * 32;
  const float r = std::min(float(target) / W, float(target) / H);
  const int nw = int(std::round(W * r)), nh = int(std::round(H * r));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh));
  if (flip) cv::flip(resized, resized, 1);
  cv::Mat canvas(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
  const int dx = (target - nw) / 2, dy = (target - nh) / 2;
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  ncnn::Mat in = ncnn::Mat::from_pixels(canvas.data, ncnn::Mat::PIXEL_BGR2RGB,
                                        target, target);
  const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(nullptr, norm);

  ncnn::Extractor ex = net_->create_extractor();
  ex.input("in0", in);
  ncnn::Mat out;
  ex.extract("out0", out);

  // out: (4 + nclases) x anchors
  std::vector<Box> boxes;
  const int nc = out.h - 4;
  for (int i = 0; i < out.w; ++i) {
    int best = 0; float bc = 0.f;
    for (int c = 0; c < nc; ++c) {
      const float v = out.row(4 + c)[i];
      if (v > bc) { bc = v; best = c; }
    }
    if (bc < 0.005f) continue;
    float cx = out.row(0)[i], cy = out.row(1)[i];
    const float w = out.row(2)[i], h = out.row(3)[i];
    if (flip) cx = target - cx;             // deshacer el flip
    Box b{int((cx - w / 2 - dx) / r), int((cy - h / 2 - dy) / r),
          int((cx + w / 2 - dx) / r), int((cy + h / 2 - dy) / r), bc};
    b.cls = best;
    boxes.push_back(b);
  }
  return boxes;
}

std::vector<Box> Detector::detect(const cv::Mat& bgr, int imgsz, bool tta) const {
  std::vector<Box> all;
  if (!net_ || bgr.empty()) return all;
  for (const Aug& a : kAugs) {
    auto v = detectOnce(bgr, imgsz, a.scale, a.flip);
    all.insert(all.end(), v.begin(), v.end());
    if (!tta) break;
  }
  // NMS por clase sobre la unión de las pasadas. Con umbral 0.005 las tres
  // pasadas juntan cientos de cajas: ordenar por (clase, conf) deja cada clase
  // contigua y el loop interno corta al cambiar de clase.
  std::sort(all.begin(), all.end(), [](const Box& a, const Box& b) {
    return a.cls != b.cls ? a.cls < b.cls : a.conf > b.conf;
  });
  std::vector<Box> keep;
  std::vector<bool> dead(all.size(), false);
  for (size_t i = 0; i < all.size(); ++i) {
    if (dead[i]) continue;
    keep.push_back(all[i]);
    for (size_t j = i + 1; j < all.size() && all[j].cls == all[i].cls; ++j)
      if (!dead[j] && iou(all[i], all[j]) > 0.45f) dead[j] = true;
  }
  std::sort(keep.begin(), keep.end(),
            [](const Box& a, const Box& b) { return a.conf > b.conf; });
  return keep;
}

}  // namespace piu
