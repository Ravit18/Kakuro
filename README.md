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

Resultados (semillas distintas de las de entrenamiento). CNN: 98.6 % de acierto en validación con 63 078 recortes de 5 estilos.

| Conjunto | OCR | Imgs | Grid | Celdas | Pistas | Tablero exacto | Resuelto |
|---|---|---|---|---|---|---|---|
| `test_set/digital` (5 estilos) | CNN | 7 | 100 % | 100 % | 100 % | 100 % | 100 % |
| `test_set/digital` | Tesseract | 7 | 100 % | 100 % | 97.4 % | 42.9 % | 42.9 % |
| `test_set/simulated` | CNN | 10 | 100 % | 100 % | 100 % | 100 % | 100 % |
| `test_set/simulated` | Tesseract | 10 | 100 % | 100 % | 86.7 % | 30 % | 30 % |
| Sintético aleatorio (estrés, 5 estilos) | CNN | 100 | 90 % | 89.4 % | 84.0 % | 63 % | 63 % |
| Sintético aleatorio (estrés, 5 estilos) | Tesseract | 100 | 90 % | 89.4 % | 52.1 % | 11 % | 11 % |

**Imágenes reales** (12 de internet y capturas de app, 6 estilos distintos; una no es un Kakuro válido):

| OCR | Tableros válidos leídos sin avisos y con solución única |
|---|---|
| CNN | **11 / 11** |
| Tesseract | 1 / 11 |

Tiempo por imagen: ~0.3 s con la CNN y ~2 s con Tesseract (CPU).

## Dataset de prueba (`data/test_set/`)

`python -m scripts.make_test_set` lo regenera:

- `digital/` (7): render limpio, tablero grande, estilo gris, captura reducida con JPEG, estilo app invertido, triángulo blanco/negro y color.
- `simulated/` (10): una condición por foto (luz baja, sombra, reflejo, ángulo leve o fuerte, rotación, desenfoque, fondo oscuro, JPEG fuerte, luz cálida).
- `printables/kakuro_para_imprimir.pdf` (8 tableros A4) con su JSON. **Imprímelos, fotografíalos con el celular** y guarda las fotos en `real/` siguiendo `real/LEEME.txt`. Es la parte "impresa" que exige el enunciado y solo la pueden hacer ustedes.

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
