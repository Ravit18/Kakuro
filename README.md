# Kakuro CP: visión computacional + programación con restricciones

Pipeline completo (end-to-end): recibe la **foto de un Kakuro**, extrae la grilla y las pistas con visión computacional y una CNN propia, y lo resuelve con **OR-Tools CP-SAT**.

```
imagen ─► preprocesamiento ─► grilla ─► celdas ─► OCR (CNN) ─► JSON ─► modelo CP ─► solución
         (OpenCV)            (proyecciones) (blanca/negra/pista) (PyTorch)  (contrato)  (CP-SAT)   (sobre la foto)
```

## Instalación

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Tesseract es opcional y solo sirve como línea base para comparar.

## Uso rápido

```bash
# Imagen -> solución (guarda en data/outputs/<nombre>/)
python main.py data/images/mi_foto.jpg --debug

# Solo la Fase 1 (genera el JSON del estado inicial)
python main.py data/images/mi_foto.jpg --only-extract

# Otro motor de OCR o tamaño forzado
python main.py foto.jpg --engine tesseract
python main.py foto.jpg --rows 9 --cols 9

# Desde un JSON (solo la Fase 2)
python main.py data/ground_truth/gen_7x7_00.json --unique
```

Salidas en `data/outputs/<nombre>/`:

| Archivo | Contenido |
|---|---|
| `puzzle.json` | Estado inicial extraído (formato en `core/schema.json`) |
| `solution.png` | Solución dibujada sobre la foto original (homografía inversa) |
| `debug/1_gray … 7_digits.png` | Etapas intermedias, útiles para el informe |

## Fase 1: visión e IA

| Paso | Módulo | Técnica |
|---|---|---|
| 1. Preprocesamiento | `vision/preprocess.py` | Escala de grises, desenfoque gaussiano y binarización adaptativa. El tablero es la componente conexa con más tinta; se toman sus 4 esquinas (envolvente convexa + `approxPolyDP`) y se corrige la perspectiva con `warpPerspective`. Si el tablero toca el borde de la imagen (captura recortada) se usa la imagen completa. El tablero rectificado se pasa a grises por **PCA del color** (máximo contraste: distingue p. ej. amarillo claro de amarillo oscuro) |
| 2. Segmentación | `vision/grid.py` | Perfiles de líneas (trazos oscuros + bordes Sobel, filtrados con apertura morfológica alargada). Se buscan conjuntamente inicio, fin y n.º de celdas que maximizan *líneas en picos − máximo en el interior de las celdas*; descarta márgenes, sombras y marcos. Las franjas claras sin bordes (papel) se recortan |
| 3. Clasificación de celdas | `vision/cells.py` | Mediana por celda y Otsu (claro/oscuro). El color "a rellenar" se deduce con una regla del Kakuro: la 1.ª fila y la 1.ª columna nunca se rellenan, así que el color de sus celdas lisas es el de las pistas (funciona con estilos invertidos de apps). Cada triángulo detecta su propio fondo y polaridad del texto; se quitan líneas rectas largas y la diagonal (continua o punteada) y se segmentan los dígitos por componentes conexas |
| 4. OCR | `vision/ocr.py` | CNN propia (tipo LeNet/VGG, 28×28) en PyTorch. Alternativa: Tesseract `--psm 7` con lista blanca 0-9 |
| 5. Estructura | `vision/pipeline.py` | JSON `{"rows","cols","grid"}`. Si una suma leída es imposible para su corrida, se corrige con la siguiente lectura más probable de la CNN que sí sea factible |

### Datos y entrenamiento

No existe un dataset público de fotos de Kakuro etiquetadas, así que el modelo se entrena con **datos sintéticos** (*domain randomization*):

1. `render/synth.py` genera tableros aleatorios válidos, los dibuja con 10 fuentes libres (`assets/fonts/`), **5 estilos** (clásico negro, gris, app invertido, triángulo blanco/negro, color), diagonal continua o punteada, sombra, recorte sin margen y tamaños variables, y simula una foto: hoja sobre un fondo, perspectiva, rotación, luz no uniforme, desenfoque, ruido y JPEG.
2. `scripts/train_digits.py` pasa cada imagen por **el mismo pipeline de inferencia** y etiqueta automáticamente cada recorte de dígito (la suma real se conoce).
3. Entrena la CNN y guarda `vision/models/digits_cnn.pt`. El modelo ya viene entrenado, así que no necesitas reentrenar para usarlo.

```bash
python -m scripts.train_digits                        # regenera/usa la caché data/digits/digits.npz
python -m scripts.train_digits --puzzles 4000 --epochs 20 --regen
python -m scripts.train_digits --real-dir data/images/train   # + tus fotos etiquetadas (fine-tuning)
```

### Fotos reales (recomendado para el informe)

1. Exporta tableros limpios para imprimir: `python -m scripts.render_puzzles --from-json data/ground_truth --clean --out data/printables`. También sirven capturas de sitios de Kakuro, citando la fuente.
2. Fotografíalos con distintos ángulos y luces. Guarda cada foto en `data/images/` junto a un JSON con el mismo nombre (`foto1.jpg` + `foto1.json` con `"grid"`). Para los impresos, basta con copiar el JSON de `data/printables`.
3. Evalúa: `python -m scripts.benchmark vision --dir data/images`

Usa esas fotos como **conjunto de prueba**. Si tienes muchas, separa algunas en `data/images/train` para afinar el modelo con `--real-dir`.

## Evaluación

```bash
python -m scripts.benchmark vision --synthetic 100                     # CNN
python -m scripts.benchmark vision --synthetic 100 --engine tesseract  # línea base
python -m scripts.benchmark solver                                     # AllDifferent+suma vs. tabla
pytest                                                                 # tests de solver y visión
```

Resultados con la CNN (semillas distintas de las de entrenamiento). Precisión en validación: 98.6 % con 63 078 recortes de los 5 estilos sintéticos.

| Conjunto | Imgs | Grid | Celdas | Pistas | Tablero exacto | Resuelto | Tiempo medio |
|---|---|---|---|---|---|---|---|
| `test_set/digital` | 4 | 100 % | 100 % | 100 % | 100 % | 100 % | 6.0 s |
| `test_set/simulated` | 10 | 100 % | 100 % | 100 % | 100 % | 100 % | 6.1 s |
| `test_set/printables` | 8 | 100 % | 100 % | 100 % | 100 % | 100 % | 5.3 s |
| `test_set/real` (capturas de app) | 11 | 100 % | 100 % | 100 % | 100 % | 100 % | 0.3 s |
| Sintético aleatorio (estrés, 5 estilos) | 100 | 90 % | 89.6 % | 82.9 % | 65 % | 65 % | 2.6 s |

Las capturas de app tienen solución única y se aceptan en la primera orientación probada. Los tableros generados por
`make_test_set` tienen más de una solución, así que el pipeline prueba las 4 orientaciones y tarda más.

Tesseract (versión anterior del pipeline, sin búsqueda de orientación ni reparación con el solver): 86.7 % de pistas en
`simulated` y 1/11 capturas de app sin errores. Para volver a medirlo: `--engine tesseract`.

## Dataset de prueba (`data/test_set/`, 33 imágenes)

- `digital/` (4): render limpio 5×5 y 10×10, estilo gris 8×8 y captura reducida al 55 % con JPEG (7×7).
- `simulated/` (10): una condición por foto (luz baja, sombra, reflejo, ángulo leve o fuerte, rotación, desenfoque, fondo oscuro, JPEG fuerte, luz cálida).
- `printables/` (8): páginas A4 de `kakuro_para_imprimir.pdf` con su JSON (estilos clásico y gris, 5×5 a 10×10).
- `real/` (11): capturas de pantalla de una aplicación de Kakuro en el celular, de 7×7 a 16×14 y 6 estilos visuales.

`python -m scripts.make_test_set` regenera `digital/`, `simulated/` y `printables/`.

## Estructura

```
core/       puzzle.py (modelo de datos + validación), schema.json (contrato JSON)
vision/     preprocess.py, grid.py, cells.py, ocr.py, pipeline.py, models/digits_cnn.pt
render/     synth.py (generador sintético), overlay.py (solución sobre la foto)
solver/     cp_model.py (AllDifferent + suma), cp_table.py (AllowedAssignments), combos.py
scripts/    generate_puzzles.py, render_puzzles.py, train_digits.py, benchmark.py
tests/      test_solver.py, test_vision.py
assets/     fonts/ (Liberation, DejaVu, FreeSans, Poppins; licencias libres OFL/GPL+FE)
```

## Referencias

- Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*, 86(11).
- Smith, R. (2007). An overview of the Tesseract OCR engine. *ICDAR 2007*.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. SMC*, 9(1).
- Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS 2017*.
- Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
- Perron, L., & Furnon, V. OR-Tools (CP-SAT). Google. https://developers.google.com/optimization
