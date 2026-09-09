# Test de paridad: el `.so` contra el pipeline Python

Corre el **mismo C++ que va al AAR** (compilado como binario Linux) sobre las
58 fotos de `piu_ocr/dataset_v2` y lo compara con `ResultReader` de Python y
con el ground truth. Es el test de regresión de cualquier cambio en el módulo.

```bash
# una vez: prebuilts Linux de ncnn + opencv-mobile (mismas versiones que el .so)
tools/host/fetch.sh

# OCR con las cajas de PyTorch (aísla el port), lista fotos que difieren
python3 tools/parity/parity.py --diff

# + detector NCNN end-to-end, regraba fixture y baseline
python3 tools/parity/parity.py --detect --fixture --update-baseline

# lo que corre antes de un commit: falla si alguna métrica cae
python3 tools/parity/parity.py --detect --baseline
gradle testReleaseUnitTest          # Kotlin interpret() == réplica Python, 90 filas
```

Piezas:

| qué | dónde |
|---|---|
| binario de host (`piuocr_cli`) | `tools/host/` → `build_host/piuocr_cli`. Usa `src/main/cpp/pipeline.cpp`, que es lo que llama `nativeRead` |
| comparador | `tools/parity/parity.py` (se relanza solo con el venv de `piu_yolo` si falta cv2) |
| métricas de referencia | `tools/parity/baseline.json` |
| fixture para Kotlin | `src/test/resources/parity_fixture.json` → `src/test/kotlin/.../ParityTest.kt` |

Qué mide: por campo (song / level / chart_type), cobertura, precisión y
acierto contra GT para Python y para nativo; acuerdo nativo == python por
campo y sobre el texto crudo; recall del detector NCNN contra las cajas de
PyTorch+TTA; latencia nativa en host.

Diferencias que quedan **a propósito** (nativo ≠ Python, medidas contra GT):

- `level`: nativo fuerza 2 dígitos en la bolita (`segmentBadge(..., 2)`), Python
  no. Nativo 0.872 de precisión vs 0.795.
- texto crudo: ~29 % de las fotos difieren en un carácter (empates en el
  producto punto con float32 en orden distinto). El match al catálogo lo absorbe:
  acuerdo de canción 0.978.

## En device

```bash
tools/parity/device.sh        # teléfono arm64 por adb; JAVA_HOME/ANDROID_HOME como para assembleRelease
```

Instala `piu-ocr-debug-androidTest.apk`, empuja las 45 fotos con GT a
`/sdcard/Android/data/com.piu.ocr.test/files/piu_parity/`, corre
`DeviceParityTest` (el `PiuOcr` real: `.so` arm64 + Kotlin) y baja
`results.json` a `build/device_results.json`. `parity.py --from-device` reporta:

- acierto end-to-end contra GT, comparado con `baseline.native_e2e` (±1 foto);
- Kotlin en device vs réplica Python sobre el mismo JSON (debe ser 0);
- `.so` arm64 vs CLI de host foto por foto (misma lógica, otra CPU: lo que
  difiera es fp16/orden de flotantes, no un bug — salvo que sea mucho);
- latencia real por foto.
