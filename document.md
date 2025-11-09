# Transformada de Hough en CUDA — Base de Informe Final

## 1. Introducción
El proyecto implementa la Transformada de Hough para detección de líneas sobre GPU usando tres configuraciones de memoria (global, constante y compartida). Además de validar la exactitud frente a una versión CPU, se busca comparar el impacto de cada jerarquía de memoria en el tiempo de ejecución del kernel y documentar el flujo completo para el informe final requerido por el curso.

## 2. Modelo teórico y transformación de coordenadas
- **Centrado del sistema:** Se traslada el origen al centro de la imagen para mantener simetría angular: `xCoord = i - w/2` y `yCoord = h/2 - j`. Esto garantiza que líneas con pendientes opuestas produzcan valores `r` equidistantes y reduce aliasing en el acumulador.
- **Inversión de eje Y:** Las imágenes PGM tienen `(0,0)` en la esquina superior izquierda. Al invertir `y`, se alinea el sistema con la convención matemática (Y positivo hacia arriba), lo cual evita que líneas con misma pendiente caigan en bins diferentes.
- **Cálculo de r:** Se usa `r(θ) = x·cos θ + y·sin θ`. Para discretizar se define `degreeBins = 90`, `θ ∈ [0,180)`, `rBins = 100`, `rMax = sqrt(w² + h²) / 2` y `rScale = 2·rMax / rBins`. El índice de acumulador es `rIdx = round((r + rMax) / rScale)` con clamp `[0, rBins-1]`.

## 3. Arquitectura CUDA
### 3.1 Flujo general (para todas las versiones)
1. Lectura de `imagen.pgm` y generación de la referencia CPU (`CPU_HoughTran`).
2. Precálculo de tablas seno/coseno en host (`computeSinCosTable`).
3. Copia de imagen, acumulador (cero) y tablas a GPU.
4. Lanzamiento del kernel seleccionado y verificación contra CPU.
5. Normalización del acumulador y dibujo de líneas detectadas (`saveAccumulatorAsPGM`, `drawDetectedLines`).

### 3.2 Versión 1 — Memoria Global
- Un hilo por píxel (`blockDim=256`).
- Tablas seno/coseno y acumulador en memoria global.
- Cada hilo realiza 90 `atomicAdd` sobre el acumulador global → mayor contención pero implementación más simple.

### 3.3 Versión 2 — Memoria Constante
- Tablas seno/coseno se declaran como `__constant__` y se copian con `cudaMemcpyToSymbol`.
- El acceso broadcast minimiza tráfico DRAM cuando muchos hilos consultan el mismo ángulo.
- Diagrama sugerido: host copia tablas → caché constante → todos los hilos leen en paralelo (pendiente de agregar al informe final).

### 3.4 Versión 3 — Memoria Compartida
- Cada bloque reserva `__shared__ int localAcc[degreeBins * rBins]` (≈36 KB).
- Fase 1: los hilos inicializan el acumulador local en paralelo.
- Fase 2: cada hilo vota en `localAcc` usando `atomicAdd` de baja latencia.
- Fase 3: reducción hacia memoria global solo para bins con conteo > 0.
- Diagrama sugerido: pipeline «Global → Shared → Global» con dos barreras (`__syncthreads()`), por agregar en la versión final.

## 4. Metodología experimental
- **Compilación:** `/usr/local/cuda/bin/nvcc -O3 houghBase.cu -o hough`.
- **Hardware:** NVIDIA GeForce GTX 1660 (driver 575.64.04, CUDA 12.9) bajo WSL2.
- **Entrada:** `imagen.pgm` (800×600). Cada corrida recalcula la referencia CPU para asegurar integridad del acumulador.
- **Comandos:** `./hough imagen.pgm v`, donde `v ∈ {1,2,3}` corresponde a Global, Constante y Compartida respectivamente.
- **Repeticiones:** 10 corridas por versión (≥ requisito de la bitácora).
- **Métrica:** “Tiempo de kernel” reportado por eventos CUDA (ms). Datos completos en `bitacora_benchmarks.md`.

## 5. Resultados de benchmark

| Versión | Memoria        | Promedio (ms) | Min (ms) | Max (ms) | Desv. Est. (ms) |
|--------:|----------------|--------------:|---------:|---------:|----------------:|
| 1       | Global         | 2.207         | 1.890    | 2.551    | 0.214 |
| 2       | Constante      | 2.362         | 1.981    | 3.315    | 0.417 |
| 3       | Compartida     | 1.894         | 1.628    | 2.438    | 0.301 |

Observaciones clave:
- La versión compartida reduce ≈14% el promedio respecto a la baseline global, gracias a la agregación local en memoria on-chip.
- La versión constante no supera consistentemente a la global: la tabla pequeña (<1 KB) cabe en caché L2/L1, por lo que el beneficio de memoria constante es marginal y sensible a latencias de `atomicAdd`.
- La desviación más grande se da en la versión constante (corrida 7 alcanzó 3.315 ms), posiblemente por variaciones térmicas o por competencia a nivel SM cuando la caché constante no estaba caliente.

## 6. Análisis de memorias especializadas
### 6.1 Memoria Constante
- Ventaja: un único fetch satisface hasta 32 hilos cuando todos consultan el mismo índice `tIdx`. Esto reduce lecturas globales para las tablas trigonométricas.
- Limitación observada: el kernel sigue realizando `atomicAdd` sobre memoria global para el acumulador, que domina el costo. Además, la tabla ya reside en caches L1/L2, por lo que la mejora es pequeña en esta imagen.
- Diagrama propuesto para el informe: Host (tablas) → `cudaMemcpyToSymbol` → Caché de Memoria Constante → SMs → Acumulador global.

### 6.2 Memoria Compartida
- Ventaja principal: amortiza los `atomicAdd` globales al realizar primero la votación dentro de `localAcc`. Solo bins con conteos no nulos disparan escrituras globales.
- Requisitos: `degreeBins * rBins * 4 B = 36 KB` < límite por bloque (≈48 KB en la GTX 1660), por lo que cada bloque consume la mayor parte de la memoria compartida disponible y limita la ocupación a 1 bloque/SM. Aun así, la reducción de tráfico global compensa.
- Para el informe final agregar diagrama con tres fases (init, vote, flush) y señalar las dos barreras.

## 7. Conclusiones preliminares y próximos pasos
1. **Validación:** Las tres versiones reproducen exactamente el acumulador CPU, lo cual habilita el análisis en el informe sin preocupaciones de precisión.
2. **Rendimiento:** Shared Memory ofrece la mejor relación costo/beneficio; Constant Memory requiere un acumulador menos contencioso para destacar.
3. **Documentación pendiente:** agregar diagramas solicitados, ampliar la explicación teórica con figuras y trasladar esta base a LaTeX/Word para el formato UVG.
4. **Trabajo futuro:** integrar perfiles (`nvprof`, Nsight Compute) para cuantificar transacciones globales/constantes y justificar con métricas.

## 8. Referencias
1. R. C. Gonzalez & R. E. Woods, *Digital Image Processing*, 4/e, Pearson, 2018.
2. R. O. Duda & P. E. Hart, “Use of the Hough Transformation to Detect Lines and Curves in Pictures,” *Communications of the ACM*, 15(1), 1972.
3. NVIDIA, *CUDA C Programming Guide*, v12.9, 2024. Disponible en: https://docs.nvidia.com/cuda/
