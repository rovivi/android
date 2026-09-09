#!/usr/bin/env bash
# Baja los prebuilts Linux de ncnn y opencv-mobile (mismas versiones que el
# .so) a third_party/host. Se corre una vez.
set -euo pipefail
cd "$(dirname "$0")/../../third_party" && mkdir -p host && cd host
NCNN=ncnn-20240820-ubuntu-2404
OCV=opencv-mobile-4.13.0-ubuntu-2404
[ -d $NCNN ] || { curl -sL -o $NCNN.zip https://github.com/Tencent/ncnn/releases/download/20240820/$NCNN.zip && unzip -q $NCNN.zip; }
[ -d $OCV ]  || { curl -sL -o $OCV.zip  https://github.com/nihui/opencv-mobile/releases/download/v36/$OCV.zip && unzip -q $OCV.zip; }
echo "ok: $(pwd)"
