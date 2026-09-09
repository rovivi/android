// Puerto de piu_ocr/recognize.py — clasificación de glifos por plantillas.
// Guardar TODOS los glifos son 97 MB de float32 y 33 K productos punto por
// carácter. 32 centroides k-means por clase puntúan igual en LOSO (0.687 vs
// 0.685) a 24x menos tamaño, y en int8 quedan en 0.41 MB: eso es lo que se
// embarca. La cuantización se hace en build_mobile.py, acá solo se lee.
#include "piu_ocr.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace piu {

namespace {
struct Header { char magic[4]; int32_t k, dim, quantized; float scale; };
}  // namespace

Templates Templates::load(const std::string& path) {
  Templates t;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return t;
  Header h{};
  if (std::fread(&h, sizeof(h), 1, f) != 1 || std::memcmp(h.magic, "PIUT", 4)) {
    std::fclose(f); return t;
  }
  std::vector<int32_t> cls(h.k);
  std::fread(cls.data(), sizeof(int32_t), h.k, f);
  t.classes_.assign(cls.begin(), cls.end());
  t.t_ = cv::Mat(h.k, h.dim, CV_32F);
  if (h.quantized) {
    std::vector<int8_t> q(size_t(h.k) * h.dim);
    std::fread(q.data(), 1, q.size(), f);
    for (int i = 0; i < h.k; ++i) {
      float* row = t.t_.ptr<float>(i);
      double n = 0.0;
      for (int j = 0; j < h.dim; ++j) {
        row[j] = q[size_t(i) * h.dim + j] * (h.scale / 127.f);
        n += double(row[j]) * row[j];
      }
      // Renormalizar: la cuantización corre cada plantilla de la esfera unidad
      // y predict compara productos punto crudos.
      n = std::sqrt(n);
      if (n > 0) for (int j = 0; j < h.dim; ++j) row[j] /= float(n);
    }
  } else {
    std::fread(t.t_.data, sizeof(float), size_t(h.k) * h.dim, f);
  }
  std::fclose(f);
  return t;
}

// Similitud coseno de cada glifo contra cada plantilla, colapsada por clase.
static cv::Mat perClass(const cv::Mat& t, const std::vector<int>& classes,
                        const std::vector<Glyph>& gs, std::vector<int>* uniq) {
  const int dim = t.cols;
  cv::Mat X(int(gs.size()), dim, CV_32F);
  for (size_t i = 0; i < gs.size(); ++i) {
    float* row = X.ptr<float>(int(i));
    double n = 0.0;
    for (int j = 0; j < dim; ++j) { row[j] = gs[i].px[j]; n += double(row[j]) * row[j]; }
    n = std::sqrt(n);
    if (n > 0) for (int j = 0; j < dim; ++j) row[j] /= float(n);
  }
  cv::Mat sims = X * t.t();

  *uniq = classes;
  std::sort(uniq->begin(), uniq->end());
  uniq->erase(std::unique(uniq->begin(), uniq->end()), uniq->end());
  cv::Mat out(sims.rows, int(uniq->size()), CV_32F, cv::Scalar(-1.f));
  for (int c = 0; c < int(uniq->size()); ++c)
    for (int k = 0; k < t.rows; ++k)
      if (classes[k] == (*uniq)[c])
        for (int i = 0; i < sims.rows; ++i)
          out.at<float>(i, c) = std::max(out.at<float>(i, c), sims.at<float>(i, k));
  return out;
}

void Templates::predict(const std::vector<Glyph>& gs, std::vector<int>* labels,
                        std::vector<float>* margins) const {
  labels->clear(); margins->clear();
  if (gs.empty() || t_.empty()) return;
  std::vector<int> uniq;
  cv::Mat pc = perClass(t_, classes_, gs, &uniq);
  for (int i = 0; i < pc.rows; ++i) {
    // El margen se toma sobre el mejor POR CLASE: con ejemplares el segundo
    // vecino suele ser de la misma clase, así que un margen sobre vecinos
    // crudos daría siempre ~0 y sería inútil como señal de confianza.
    float b1 = -2.f, b2 = -2.f; int bi = 0;
    for (int c = 0; c < pc.cols; ++c) {
      const float v = pc.at<float>(i, c);
      if (v > b1) { b2 = b1; b1 = v; bi = c; }
      else if (v > b2) b2 = v;
    }
    labels->push_back(uniq[bi]);
    margins->push_back(b1 - std::max(b2, 0.f));
  }
}

void Templates::scores(const std::vector<Glyph>& gs,
                       std::vector<std::vector<float>>* out) const {
  out->clear();
  if (gs.empty() || t_.empty()) return;
  std::vector<int> uniq;
  cv::Mat pc = perClass(t_, classes_, gs, &uniq);
  for (int i = 0; i < pc.rows; ++i) {
    // Indexado por dígito 0..9; las clases ausentes quedan en -1 para que nunca
    // ganen. Lo necesita el cruce con el catálogo, que reordena niveles enteros
    // y no solo acepta o rechaza el argmax.
    std::vector<float> row(10, -1.f);
    for (int c = 0; c < pc.cols; ++c)
      if (uniq[c] >= 0 && uniq[c] < 10) row[uniq[c]] = pc.at<float>(i, c);
    out->push_back(std::move(row));
  }
}

}  // namespace piu
