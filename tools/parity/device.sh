#!/usr/bin/env bash
# Paridad EN DEVICE: instala el test instrumentado, empuja las fotos del
# dataset, corre el módulo real (arm64 + Kotlin) y compara el resultado con
# host, Python y ground truth.
#
#   tools/parity/device.sh            # teléfono conectado por adb
#
# Necesita JAVA_HOME (17/21) y ANDROID_HOME como para assembleRelease.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${JAVA_HOME:=$HOME/.jdks/ms-17.0.20.1}"; : "${ANDROID_HOME:=$HOME/Android/Sdk}"
export JAVA_HOME ANDROID_HOME
GRADLE=${GRADLE:-$(ls -d ~/.gradle/wrapper/dists/gradle-8.13-bin/*/gradle-8.13/bin/gradle | head -1)}
ADB=${ADB:-$ANDROID_HOME/platform-tools/adb}
PKG=com.piu.ocr.test
DEV_DIR=/sdcard/Android/data/$PKG/files/piu_parity
OUT=${OUT:-build/device_results.json}

$ADB get-state >/dev/null || { echo "no hay device por adb"; exit 1; }
echo "== build + install del test"
"$GRADLE" -q assembleDebugAndroidTest
$ADB install -r -t build/outputs/apk/androidTest/debug/*.apk >/dev/null

echo "== fotos"
$ADB shell mkdir -p "$DEV_DIR"
python3 - <<'PY' | while read -r f; do $ADB push -q "../DATASET/$f" "$DEV_DIR/" ; done
import json
b=json.load(open('../dataset_v2/boxes.json')); g=json.load(open('../dataset_v2/gt_song.json'))
for r in b:
    if not r['key'].startswith('_') and g.get(r['key']): print(r['file'])
PY

echo "== corriendo en device"
$ADB shell am instrument -w -e class com.piu.ocr.DeviceParityTest \
    $PKG/androidx.test.runner.AndroidJUnitRunner | tail -3
$ADB pull -q "$DEV_DIR/results.json" "$OUT"
echo "== comparando ($OUT)"
python3 tools/parity/parity.py --from-device "$OUT"
