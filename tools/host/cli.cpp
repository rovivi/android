// CLI de host: corre piu::Engine sobre una imagen y escribe JSON en stdout.
//
//   piuocr_cli --assets DIR IMG [--box cls,x1,y1,x2,y2,conf ...]
//
// Sin --box corre el detector YOLO (con TTA, como el teléfono). Con --box usa
// esas cajas y saltea el detector, para aislar el OCR de la detección.
// Salida: {"result": <mismo JSON que nativeRead>, "boxes": [...], "ms": {...}}
#include "piu_ocr.h"
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace piu;

static double ms(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - a).count();
}

int main(int argc, char** argv) {
  std::string assets, image;
  std::vector<Box> given;
  bool useGiven = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--assets") && i + 1 < argc) assets = argv[++i];
    else if (!std::strcmp(argv[i], "--box") && i + 1 < argc) {
      Box b{};
      if (std::sscanf(argv[++i], "%d,%d,%d,%d,%d,%f", &b.cls, &b.x1, &b.y1,
                      &b.x2, &b.y2, &b.conf) != 6) {
        std::fprintf(stderr, "--box mal formado: %s\n", argv[i]);
        return 2;
      }
      given.push_back(b);
      useGiven = true;
    } else if (argv[i][0] != '-') image = argv[i];
  }
  if (assets.empty() || image.empty()) {
    std::fprintf(stderr, "uso: piuocr_cli --assets DIR IMG [--box cls,x1,y1,x2,y2,conf ...]\n");
    return 2;
  }

  auto t0 = std::chrono::steady_clock::now();
  Engine eng;
  if (!eng.load(assets)) {
    std::fprintf(stderr, "no se pudieron cargar los assets de %s\n", assets.c_str());
    return 1;
  }
  const double loadMs = ms(t0);

  cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::fprintf(stderr, "no se pudo leer %s\n", image.c_str());
    return 1;
  }

  std::vector<Box> used;
  t0 = std::chrono::steady_clock::now();
  const std::string result = eng.read(img, useGiven ? &given : nullptr, &used);
  const double readMs = ms(t0);

  std::printf("{\"result\":%s,\"boxes\":[", result.c_str());
  for (size_t i = 0; i < used.size(); ++i)
    std::printf("%s{\"cls\":%d,\"box\":[%d,%d,%d,%d],\"conf\":%g}", i ? "," : "",
                used[i].cls, used[i].x1, used[i].y1, used[i].x2, used[i].y2,
                used[i].conf);
  std::printf("],\"ms\":{\"load\":%.1f,\"read\":%.1f},\"detected\":%s}\n",
              loadMs, readMs, useGiven ? "false" : "true");
  return 0;
}
