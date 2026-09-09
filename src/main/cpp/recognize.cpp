// Puerto de piu_ocr/recognize.py — clasificación de glifos por plantillas.
// Guardar TODOS los glifos son 97 MB de float32 y 33 K productos punto por
// carácter. 32 centroides k-means por clase puntúan igual en LOSO (0.687 vs
// 0.685) a 24x menos tamaño, y en int8 quedan en 0.41 MB: eso es lo que se
// embarca. La cuantización se hace en build_mobile.py, acá solo se lee.
#include "piu_ocr.h"
#include <cstdint>
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
  // Cada fread se verifica: un archivo truncado antes devolvía plantillas
  // llenas de basura y predict seguía respondiendo, solo que cualquier cosa.
  bool ok = std::fread(&h, sizeof(h), 1, f) == 1 && !std::memcmp(h.magic, "PIUT", 4)
            && h.k > 0 && h.k < (1 << 20) && h.dim == GLYPH_W * GLYPH_H;
  std::vector<int32_t> cls;
  cv::Mat m;
  if (ok) {
    cls.resize(h.k);
    ok = std::fread(cls.data(), sizeof(int32_t), h.k, f) == size_t(h.k);
  }
  if (ok) {
    m.create(h.k, h.dim, CV_32F);
    if (h.quantized) {
      std::vector<int8_t> q(size_t(h.k) * h.dim);
      ok = std::fread(q.data(), 1, q.size(), f) == q.size();
      const float s = h.scale / 127.f;
      for (int i = 0; ok && i < h.k; ++i) {
        float* row = m.ptr<float>(i);
        double n = 0.0;
        for (int j = 0; j < h.dim; ++j) {
          row[j] = q[size_t(i) * h.dim + j] * s;
          n += double(row[j]) * row[j];
        }
        // Renormalizar: la cuantización corre cada plantilla de la esfera
        // unidad y predict compara productos punto crudos.
        n = std::sqrt(n);
        if (n > 0) for (int j = 0; j < h.dim; ++j) row[j] /= float(n);
      }
    } else {
      const size_t cnt = size_t(h.k) * h.dim;
      ok = std::fread(m.data, sizeof(float), cnt, f) == cnt;
    }
  }
  std::fclose(f);
  if (!ok) return t;

  t.t_ = m;
  t.classes_.assign(cls.begin(), cls.end());
  // Clases distintas y el índice de cada plantilla en esa lista, una sola vez:
  // antes se recalculaba por llamada con un triple loop K x C x N.
  t.uniq_ = t.classes_;
  std::sort(t.uniq_.begin(), t.uniq_.end());
  t.uniq_.erase(std::unique(t.uniq_.begin(), t.uniq_.end()), t.uniq_.end());
  t.clsIdx_.resize(t.classes_.size());
  for (size_t k = 0; k < t.classes_.size(); ++k)
    t.clsIdx_[k] = int(std::lower_bound(t.uniq_.begin(), t.uniq_.end(),
                                        t.classes_[k]) - t.uniq_.begin());
  return t;
}

// Similitud coseno de cada glifo contra cada plantilla, colapsada por clase
// (máximo). Sale (N, C).
cv::Mat Templates::perClass(const std::vector<Glyph>& gs) const {
  const int dim = t_.cols;
  cv::Mat X(int(gs.size()), dim, CV_32F);
  for (size_t i = 0; i < gs.size(); ++i) {
    float* row = X.ptr<float>(int(i));
    double n = 0.0;
    for (int j = 0; j < dim; ++j) { row[j] = gs[i].px[j]; n += double(row[j]) * row[j]; }
    n = std::sqrt(n);
    if (n > 0) for (int j = 0; j < dim; ++j) row[j] /= float(n);
  }
  // GEMM_2_T evita materializar t_.t() (2346 x 768 floats, 7 MB) por llamada.
  cv::Mat sims;
  cv::gemm(X, t_, 1.0, cv::noArray(), 0.0, sims, cv::GEMM_2_T);

  cv::Mat out(sims.rows, int(uniq_.size()), CV_32F, cv::Scalar(-1.f));
  for (int i = 0; i < sims.rows; ++i) {
    const float* s = sims.ptr<float>(i);
    float* o = out.ptr<float>(i);
    for (int k = 0; k < sims.cols; ++k) {
      const int c = clsIdx_[k];
      if (s[k] > o[c]) o[c] = s[k];
    }
  }
  return out;
}

void Templates::predict(const std::vector<Glyph>& gs, std::vector<int>* labels,
                        std::vector<float>* margins) const {
  labels->clear(); margins->clear();
  if (gs.empty() || t_.empty()) return;
  const std::vector<int>& uniq = uniq_;
  cv::Mat pc = perClass(gs);
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
  const std::vector<int>& uniq = uniq_;
  cv::Mat pc = perClass(gs);
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
