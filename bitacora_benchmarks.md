# Bitácora de Benchmarks CUDA

Contexto para reproducibilidad:
- Imagen de entrada: `imagen.pgm` (800×600, monocromática binaria).
- GPU: NVIDIA GeForce GTX 1660 (Driver 575.64.04, CUDA 12.9).
- Host: WSL2 sobre Windows, `nvcc` `/usr/local/cuda/bin` (O3).
- Comando base: `./hough imagen.pgm <version>`; cada corrida recalcula la referencia CPU y sobrescribe `output_v*.pgm` / `lines_v*.ppm`.
- Métrica registrada: tiempo del kernel reportado por eventos CUDA (ms). Estadísticos con desviación estándar muestral.

## Resumen estadístico (10 corridas por versión)

| Versión | Tipo de memoria    | Min (ms) | Max (ms) | Promedio (ms) | Desv. estándar (ms) |
|---------|--------------------|---------:|---------:|--------------:|--------------------:|
| 1       | Global (baseline)  | 1.890    | 2.551    | 2.207         | 0.214               |
| 2       | Constante          | 1.981    | 3.315    | 2.362         | 0.417               |
| 3       | Compartida         | 1.628    | 2.438    | 1.894         | 0.301               |

## Mediciones individuales

### Versión 1 – Memoria Global

| Corrida | Tiempo (ms) |
|--------:|-----------:|
| 1 | 2.237 |
| 2 | 2.441 |
| 3 | 2.245 |
| 4 | 1.890 |
| 5 | 2.551 |
| 6 | 1.996 |
| 7 | 2.071 |
| 8 | 2.429 |
| 9 | 2.058 |
| 10 | 2.151 |

### Versión 2 – Memoria Constante

| Corrida | Tiempo (ms) |
|--------:|-----------:|
| 1 | 1.981 |
| 2 | 2.261 |
| 3 | 2.198 |
| 4 | 1.997 |
| 5 | 2.599 |
| 6 | 2.021 |
| 7 | 3.315 |
| 8 | 2.713 |
| 9 | 2.155 |
| 10 | 2.382 |

### Versión 3 – Memoria Compartida

| Corrida | Tiempo (ms) |
|--------:|-----------:|
| 1 | 2.369 |
| 2 | 1.949 |
| 3 | 1.751 |
| 4 | 1.664 |
| 5 | 2.052 |
| 6 | 1.653 |
| 7 | 1.743 |
| 8 | 1.693 |
| 9 | 2.438 |
| 10 | 1.628 |
