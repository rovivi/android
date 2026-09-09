# piu-ocr Android — informe de estado

Lectura de pantallas de resultado de Pump It Up en el teléfono, sin LLM:
YOLO en NCNN + OCR clásico en C++ (OpenCV core/imgproc) + matching difuso
contra el catálogo en Kotlin. Un AAR de 9.4 MB, ABI `arm64-v8a`.

Este informe resume qué se midió, qué se arregló y qué falta. Todos los
números salen de `tools/parity/parity.py` sobre las **45 fotos de cabina con
ground truth** de `piu_ocr/dataset_v2` (58 en total, 45 con título legible), y
se regeneran con:

```bash
python3 tools/parity/parity.py --detect --json docs/parity_detail.json --update-baseline
python3 tools/parity/report.py
```

---

## 1. Antes y después

El módulo del primer commit **compilaba y respondía, pero no leía nada**:
el detector devolvía basura, el nivel salía como code points y ningún error
llegaba a Kotlin. Nada de eso era visible sin correr el C++ sobre fotos reales.

![antes / después](img/before_after.png)

| | antes | ahora |
|---|---|---|
| canción end-to-end (acierto) | 0 | **0.733** |
| recall de `song_name` del detector | 0 | **0.956** |
| nivel (acierto, cajas PyTorch) | 0 | **0.810** |
| acuerdo del texto crudo nativo == Python | 0.533 | 0.711 |
| Kotlin ≠ Python (de 90 filas) | 4 | **0** |
| `libpiuocr.so` | 12.84 MB | **7.61 MB** |

## 2. Paridad con el pipeline Python

Mismas fotos, mismas cajas de YOLO (PyTorch + TTA), mismas plantillas int8.
Tres columnas: la referencia Python, el C++ del `.so` con esas cajas, y el C++
end-to-end con su propio detector NCNN.

![campos](img/fields.png)

- **Canción**: el port empata con Python (0.756) y con su detector queda a 2 pts.
  Precisión cuando responde: 0.89–0.92. Lo que no pasa el gate se escala al VLM.
- **Nivel**: el port es *mejor* que Python (0.81 vs 0.74) porque fuerza dos
  dígitos en la bolita; el catálogo sabe cuántos son.
- **Chart type**: 0.84 con cajas PyTorch, 0.77 con las de NCNN — el recorte de
  la bolita es sensible a qué tan ajustada viene la caja. Es el siguiente
  candidato a medir (`pad` en `pipeline.cpp`).

Acuerdo campo por campo con Python sobre las mismas cajas, y latencia:

![latencia y acuerdo](img/latency_agreement.png)

El 29 % de las fotos difiere en **un carácter** del texto crudo (empates en el
producto punto en float32); el catálogo cerrado lo absorbe: canción 0.978 de
acuerdo.

## 3. El gate funciona

El margen entre el primer y el segundo candidato del catálogo separa aciertos
de errores. Por debajo de 0.015 la respuesta se rechaza y se escala; de las 37
que pasan, 33 son correctas.

![margen](img/margin.png)

## 4. Detector: el TTA a mano

El export a NCNN es de forma fija (`Reshape 0=33600` en el `.param`). La pasada
TTA a 1056 px devolvía cajas con confianza 0.9999 en la clase equivocada, que
ganaban el NMS. La reducción ahora se hace **adentro del lienzo de 1280**, y se
midieron 11 combinaciones de pasadas:

![TTA](img/tta.png)

`1, 0.83, 0.83f` (original, reducida, reducida con flip) cuesta lo mismo que la
combinación anterior y sube el acierto de 0.644 a 0.733. El flip sin reducir
casi no aporta; el flip **de la reducida** sí. Dos pasadas (`1, 0.83f`) dan el
mismo acierto un 26 % más barato con algo menos de precisión: es la opción si
la latencia en el teléfono no cierra.

![detector](img/detector.png)

## 5. Bugs que encontró el test de paridad

| dónde | síntoma | causa |
|---|---|---|
| `detector.cpp` | 0/58 títulos | export de forma fija; TTA a otro tamaño = basura |
| `recognize.cpp` | nivel "5054" | `build_mobile.py` guarda `'2'` como 50; `digitOf()` acepta ambos |
| `pipeline.cpp` | chart type 0.88 → 0.81 | votaba entre todas las bolitas; una caja floja sobre pantalla azul gana con conf 1.0 |
| `SongMatcher.kt` | canción distinta al borde del gate | `similarity()` era LCS y `difflib.ratio()` no lo es; Python redondea a 4 decimales; `org.json` de JVM no preserva el orden del catálogo; Kotlin filtraba por chart type y Python no; `normalize()` sin NFD |
| `text.cpp`, `segment.cpp` | glifos un píxel distintos | `round()` de Python es half-even; `std::lround` no; mediana de numpy promedia |
| `jni_bridge.cpp` | handle a medias, `"{}"` que rompía `getJSONArray`, `cv::Exception` abortaba el proceso, `lockPixels` sin unlock | robustez JNI |
| `PiuOcr.kt` | modelos viejos tras actualizar la app, double free en `close()`, bitmaps `HARDWARE` vacíos | copia de assets por versión + atómica, close idempotente, conversión a `ARGB_8888` |

## 6. Cómo se verifica cualquier cambio

```bash
tools/host/fetch.sh                                   # una vez: ncnn + opencv-mobile Linux
python3 tools/parity/parity.py --detect --baseline    # falla si alguna métrica cae
gradle testReleaseUnitTest                            # Kotlin interpret() == réplica Python, 90 filas
tools/parity/device.sh                                # lo mismo, en un teléfono por adb
```

Tres capas: el C++ como binario de host contra Python y ground truth
(`baseline.json` como gate de regresión); un fixture que `ParityTest.kt`
reproduce con el `PiuOcr.interpret()` real en la JVM; y el test instrumentado
que corre el `.so` arm64 real y compara contra las otras dos.

## 7. Lo que falta

- **Correr en device.** Todo lo de arriba es en x86; ncnn en ARM usa fp16 y
  puede mover 1–3 fotos. Latencia estimada 2–4 s por foto con 3 pasadas.
- **Chart type y nivel con cajas NCNN** (0.77 / 0.71): medir `pad` de la bolita.
- **Export dinámico del YOLO** para un TTA multi-escala real, o más fotos de
  entrenamiento en las que `song_name` sale con poca confianza.
- Latencia: el detector es el 90 % del tiempo. Cuantización int8 real
  (`ncnn2int8`, ≥300 imágenes de calibración) es la palanca grande.
