// Puerto de piu_ocr/badge.py — tipo de chart por el color de la bolita.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <array>

namespace piu {

// Rangos de tono en la convención de OpenCV (H en 0..179). El naranja del
// single cruza el 0, así que va en dos tramos.
struct HueRange { const char* name; int lo1, hi1, lo2, hi2; };
static const std::array<HueRange, 4> kHues = {{
    {"single",     0,  22, 168, 179},   // naranja / rojo
    {"coop",      23,  34,  -1,  -1},   // amarillo
    {"double",    35,  85,  -1,  -1},   // verde
    {"halfdouble", 86, 130,  -1,  -1},  // azul / cian
}};

bool classifyChartType(const cv::Mat& roi, std::string* out, float* conf) {
  if (roi.empty()) return false;
  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  cv::Mat mask = (ch[1] > 80) & (ch[2] > 60);

  // La bolita es un disco inscrito en la caja, pero las esquinas de la caja son
  // pantalla, que en el layout Phoenix es azul. Con la caja floja de YOLO esas
  // esquinas ganaban el voto y 39 de 84 bolitas salían "halfdouble".
  // Restringir el voto al disco inscrito lo arregla (0.500 -> 0.833).
  mask &= discMask(mask.rows, mask.cols, DISC_R);

  const int total = cv::countNonZero(mask);
  if (total < 20) return false;

  int bestCount = 0; const char* bestName = nullptr;
  for (const auto& r : kHues) {
    cv::Mat in = (ch[0] >= r.lo1) & (ch[0] <= r.hi1) & mask;
    int c = cv::countNonZero(in);
    if (r.lo2 >= 0) {
      cv::Mat in2 = (ch[0] >= r.lo2) & (ch[0] <= r.hi2) & mask;
      c += cv::countNonZero(in2);
    }
    if (c > bestCount) { bestCount = c; bestName = r.name; }
  }
  if (!bestName || bestCount == 0) return false;
  *out = bestName;
  *conf = float(bestCount) / float(total);
  return true;
}

}  // namespace piu
