#!/usr/bin/env python3
"""Gráficas del informe (docs/img/*.png) a partir de baseline.json y del
detalle por foto de parity.py --json.

    python3 tools/parity/parity.py --detect --json docs/parity_detail.json
    python3 tools/parity/report.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANDROID = os.path.abspath(os.path.join(HERE, "..", ".."))
DOCS = os.path.join(ANDROID, "docs")
IMG = os.path.join(DOCS, "img")
os.makedirs(IMG, exist_ok=True)

base = json.load(open(os.path.join(HERE, "baseline.json")))
detail = json.load(open(os.path.join(DOCS, "parity_detail.json")))

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C_PY, C_NAT, C_E2E, C_BAD, C_OK = "#7f8c8d", "#2980b9", "#27ae60", "#c0392b", "#27ae60"
FIELDS = ["song", "level", "chart_type"]
LABEL = {"song": "canción", "level": "nivel", "chart_type": "chart type"}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, name), bbox_inches="tight")
    plt.close()


# 1. acierto por campo: python vs nativo (cajas PyTorch) vs nativo end-to-end
def fig_fields():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, metric in zip(axes, ["accuracy", "precision", "coverage"]):
        x = np.arange(len(FIELDS)); w = 0.26
        for i, (side, key, col) in enumerate([("Python (ref.)", "python", C_PY),
                                              ("nativo, cajas PyTorch", "native", C_NAT),
                                              ("nativo end-to-end (NCNN)", "native_e2e", C_E2E)]):
            vals = [base[key][f][metric] for f in FIELDS]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=side, color=col)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[f] for f in FIELDS])
        ax.set_ylim(0, 1.08)
        ax.set_title({"accuracy": "acierto (correcto / total)",
                      "precision": "precisión (correcto / respondido)",
                      "coverage": "cobertura (respondido / total)"}[metric], fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    n = base["native"]["song"]["n"]
    fig.suptitle(f"Paridad sobre {n} fotos de cabina con ground truth", fontsize=12)
    save("fields.png")


# 2. barrido del TTA (medido con --augs)
TTA = [  # pasadas, recall song_name, acierto canción e2e, latencia host ms
    ("1", 0.733, 0.444, 276), ("1, 1f", 0.756, 0.489, 481),
    ("1, 0.83", 0.844, 0.600, 490), ("1, 0.83f", 0.933, 0.733, 486),
    ("1, 0.83, 1f\n(antes)", 0.844, 0.644, 661),
    ("1, 0.83, 0.83f\n(ahora)", 0.956, 0.733, 653),
    ("1, 0.83, 1f, 0.83f", 0.933, 0.733, 844),
    ("4 pasadas (0.9/0.83f/0.75)", 0.933, 0.711, 1006),
    ("5 pasadas", 0.933, 0.711, 1062),
]


def fig_tta():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, rec, acc, ms in TTA:
        now = "ahora" in name; before = "antes" in name
        col = C_E2E if now else (C_BAD if before else C_NAT)
        ax.scatter(ms, acc, s=90 if (now or before) else 45, color=col, zorder=3,
                   edgecolor="black" if now else "none")
        right = ms > 950
        ax.annotate(name.replace("\n", " "), (ms, acc), textcoords="offset points",
                    xytext=(-6 if right else 6, (8 if now else -3) + (10 if "5 pasadas" in name else 0)),
                    ha="right" if right else "left", fontsize=7.5)
    ax.axhline(0.756, ls="--", color=C_PY, lw=1)
    ax.text(300, 0.762, "Python + PyTorch TTA = 0.756", color=C_PY, fontsize=8)
    ax.set_xlabel("latencia por foto en host, ms (Ryzen 7 5700X, 4 hilos)")
    ax.set_ylabel("acierto de canción end-to-end")
    ax.set_ylim(0.4, 0.8)
    ax.set_title("Pasadas del TTA del detector: mismo costo, +9 pts", fontsize=11)
    save("tta.png")


# 3. detector: recall por clase
def fig_detector():
    d = base["detector"]
    names = list(d); vals = [d[k]["recall@0.5"] for k in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh(names, vals, color=[C_E2E if v >= 0.95 else C_NAT for v in vals])
    for b, v, k in zip(bars, vals, names):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}  (n={d[k]['n']})",
                va="center", fontsize=8)
    ax.set_xlim(0, 1.15); ax.invert_yaxis()
    ax.set_xlabel("recall de la mejor caja (IoU ≥ 0.5) contra PyTorch + TTA")
    ax.set_title("Detector YOLO en NCNN fp16, 3 pasadas", fontsize=11)
    save("detector.png")


# 4. antes / después de la sesión
def fig_before_after():
    items = [
        ("canción end-to-end\n(acierto)", 0.0, 0.733, "", True),
        ("recall song_name\n(detector)", 0.0, 0.956, "", True),
        ("nivel leído\n(acierto, cajas PyTorch)", 0.0, 0.810, "", True),
        ("acuerdo texto crudo\nnativo == Python", 0.533, 0.711, "", True),
        ("Kotlin ≠ Python\n(fotos de 90)", 4, 0, "", False),
        ("libpiuocr.so, MB", 12.84, 7.61, "", False),
    ]
    fig, axes = plt.subplots(1, len(items), figsize=(12.5, 3.4))
    for ax, (name, b, a, unit, higher) in zip(axes, items):
        bars = ax.bar(["antes", "ahora"], [b, a], color=[C_BAD, C_E2E], width=0.6)
        for bar, v in zip(bars, [b, a]):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:g}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_title(name, fontsize=9); ax.grid(False)
        ax.set_ylim(0, max(b, a) * 1.25 or 1)
        ax.set_yticks([])
    fig.suptitle("Estado del módulo al inicio del primer commit vs ahora", fontsize=12)
    save("before_after.png")


# 5. margen de canción: correctas vs incorrectas, y el gate
def fig_margin():
    ok, bad, rej = [], [], []
    from unicodedata import normalize as _n
    import re
    def norm(s):
        s = _n("NFD", s or ""); s = "".join(c for c in s if not (0x300 <= ord(c) <= 0x36f)).lower()
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()
    for r in detail["detect"]:
        m = r["native"]["song_margin"]; v = r["native"]["song"]; g = r["gt"]["song"]
        if v is None:
            rej.append(m)
        elif norm(v) == norm(g):
            ok.append(m)
        else:
            bad.append(m)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, 0.25, 26)
    ax.hist([ok, bad], bins=bins, stacked=True, color=[C_OK, C_BAD],
            label=[f"canción correcta ({len(ok)})", f"canción incorrecta ({len(bad)})"])
    ax.hist(rej, bins=bins, color=C_PY, alpha=0.6, label=f"rechazada por el gate ({len(rej)})")
    ax.axvline(0.015, color="black", ls="--", lw=1)
    ax.text(0.017, ax.get_ylim()[1] * 0.9, "gate MIN_SONG_MARGIN = 0.015", fontsize=8)
    ax.set_xlabel("margen entre el 1° y el 2° candidato del catálogo")
    ax.set_ylabel("fotos"); ax.legend(fontsize=8)
    ax.set_title("El gate convierte errores en escalaciones al VLM (end-to-end)", fontsize=11)
    save("margin.png")


# 6. latencia por foto y acuerdo por campo
def fig_latency_agreement():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    ms = sorted(r["ms"] for r in detail["detect"])
    a1.bar(range(len(ms)), ms, color=C_NAT, width=1.0)
    a1.axhline(np.mean(ms), color="black", ls="--", lw=1)
    a1.text(0, np.mean(ms) + 15, f"media {np.mean(ms):.0f} ms", fontsize=8)
    a1.set_xlabel("foto (ordenadas)"); a1.set_ylabel("ms")
    a1.set_title("Latencia end-to-end en host (detector 3 pasadas + OCR)", fontsize=10)
    ag = base["agreement"]
    keys = ["raw", "level", "chart_type", "song"]
    vals = [ag[k] for k in keys]
    bars = a2.barh([{"raw": "texto crudo del título"}.get(k, LABEL.get(k, k)) for k in keys], vals,
                   color=[C_PY, C_NAT, C_NAT, C_E2E])
    for b, v in zip(bars, vals):
        a2.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    a2.set_xlim(0, 1.12)
    a2.set_title("Acuerdo nativo == Python, mismas cajas", fontsize=10)
    save("latency_agreement.png")


if __name__ == "__main__":
    fig_fields(); fig_tta(); fig_detector(); fig_before_after(); fig_margin()
    fig_latency_agreement()
    print("ok:", sorted(os.listdir(IMG)))
