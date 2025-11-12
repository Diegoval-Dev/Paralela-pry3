# Contexto del Proyecto - Transformada de Hough en CUDA

Análisis completo del proyecto para referencia de implementación.

---

## 📋 INFORMACIÓN GENERAL

**Proyecto:** #3 - Transformada de Hough usando CUDA  
**Curso:** Computación Paralela y Distribuida  
**Institución:** Universidad del Valle de Guatemala  
**Fecha de Entrega:** Semana del 12-14 de noviembre  
**Grupo:** Máximo 3 personas

---

## 🎯 OBJETIVOS DEL PROYECTO

1. **Conocer** aplicación práctica de la memoria Constante de GPU
2. **Aprovechar** características de memorias Global, Compartida y Constante
3. **Implementar** algoritmo clásico de Computer Vision en arquitectura paralela

---

## 📖 DESCRIPCIÓN DEL ALGORITMO

### ¿Qué es la Transformada de Hough?
- Técnica de Computer Vision para **detectar líneas rectas** en imágenes blanco y negro
- Sistema de **votación**: cada pixel "iluminado" vota por líneas posibles a las que pertenece
- Las líneas con **más votos** representan líneas reales en la imagen

### Fórmula Principal
```
r(θ) = x·cos(θ) + y·sin(θ)
```
Donde:
- `r`: distancia del origen a la línea
- `θ`: ángulo perpendicular a la línea
- `(x,y)`: coordenadas del pixel (origen en centro de imagen)

### Parámetros de Discretización
- **θ (theta):** 90 bins, incrementos de 2°, rango [0°, 180°)
- **r (distancia):** 100 bins, rango [-rMax, rMax]
- **rMax:** `sqrt(w² + h²) / 2` (diagonal máxima desde centro)
- **rScale:** `2 * rMax / rBins`

---

## 📂 ESTRUCTURA DEL CÓDIGO BASE

### Archivos Principales
```
proyecto/
├── houghBase.cu          # Implementación principal CUDA
├── common/
│   ├── pgm.h            # Clase para leer imágenes PGM
│   └── pgm.cpp          # (compilado a pgm.o)
├── Makefile             # Configuración de compilación
├── .gitignore           # Archivos ignorados por git
└── test.cu              # Test simple de CUDA
```

### Funciones Implementadas

#### ✅ YA IMPLEMENTADO en `houghBase.cu`:

1. **`CPU_HoughTran(...)`**
   - Versión secuencial CPU de referencia
   - Calcula acumulador completo en host
   - Usado para validar resultados GPU

2. **`accumulateHoughGPU<<< >>> (...)`** ⚠️ BASELINE
   - Kernel GPU con **solo memoria Global**
   - 1 thread por pixel
   - Configuración: `blockNum = (w*h + 256 - 1) / 256` bloques de 256 threads
   - Usa `atomicAdd` para evitar race conditions
   - **gloID calculado:** `blockIdx.x * blockDim.x + threadIdx.x`

3. **`computeSinCosTable(...)`**
   - Pre-calcula valores de sin/cos en host
   - Evita operaciones trigonométricas costosas en GPU

4. **`saveAccumulatorAsPGM(...)`**
   - Guarda acumulador como imagen PGM normalizada
   - Formato: ancho=degreeBins, alto=rBins

5. **Medición de Tiempo**
   - ✅ Implementado con CUDA events
   - Mide solo tiempo de kernel (no incluye transfers)

6. **Liberación de Memoria**
   - ✅ Ya libera: `d_in, d_hough, d_Cos, d_Sin`
   - ✅ Ya libera: `h_hough, cpuht, pcCos, pcSin`

---

## ✅ ESTADO ACTUAL - CÓDIGO FUNCIONAL COMPLETADO

### **✅ IMPLEMENTACIONES COMPLETADAS:**

#### **Tarea 1: ✅ Versión Global (Baseline)**
- [x] Kernel `accumulateHoughGPU` funcionando
- [x] Usa memoria global para tablas sin/cos
- [x] Liberación de memoria implementada

#### **Tarea 4: ✅ Visualización de Líneas**
- [x] Función `drawDetectedLines()` implementada
- [x] Genera imágenes PPM (`.ppm`) con líneas rojas superpuestas
- [x] Threshold automático: `max(promedio + 2*stddev, max/4)`
- [x] Detecta y dibuja líneas con votos > threshold

#### **Tarea 5: ✅ Memoria Constante**
- [x] Variables `__constant__ float d_Cos_const[degreeBins]` declaradas
- [x] Variables `__constant__ float d_Sin_const[degreeBins]` declaradas
- [x] Kernel `GPU_HoughTranConst()` implementado
- [x] Usa `cudaMemcpyToSymbol()` para copiar a memoria constante
- [x] Kernel no requiere parámetros de tablas sin/cos

#### **Tarea 8: ✅ Memoria Compartida**
- [x] Kernel `GPU_HoughTranShared()` implementado
- [x] Acumulador local `__shared__ int localAcc[degreeBins * rBins]`
- [x] Inicialización distribuida entre threads del bloque
- [x] Dos barreras `__syncthreads()` correctamente ubicadas
- [x] Votos en acumulador local + copia final a global
- [x] Usa memoria constante para tablas sin/cos

#### **✅ CÓDIGO MODULAR:**
- [x] Función `runKernelVersion()` para ejecutar cualquier versión
- [x] Main permite ejecutar versión específica o todas: `./hough image.pgm [1|2|3|0]`
  - `1` = Solo Global Memory
  - `2` = Solo Constant Memory
  - `3` = Solo Shared Memory
  - `0` = Todas las versiones (default)
- [x] Salidas separadas por versión: `output_v1.pgm`, `lines_v1.ppm`, etc.
- [x] Verificación automática contra CPU para cada versión
- [x] Medición de tiempo con CUDA events

#### **✅ ARCHIVOS GENERADOS:**
- `output_v1.pgm` - Acumulador versión Global
- `output_v2.pgm` - Acumulador versión Constante
- `output_v3.pgm` - Acumulador versión Compartida
- `lines_v1.ppm` - Líneas detectadas versión Global
- `lines_v2.ppm` - Líneas detectadas versión Constante
- `lines_v3.ppm` - Líneas detectadas versión Compartida

---

## 📋 TAREAS PENDIENTES PARA CONTINUACIÓN

### **Benchmarking y Análisis (Tareas 2, 6, 9):**
- [ ] Ejecutar **mínimo 10 mediciones** por cada versión
- [ ] Registrar tiempos en bitácora para análisis estadístico
- [ ] Calcular promedio, desviación estándar, mínimo, máximo
- [ ] Comparar rendimiento entre las 3 versiones

### **Documentación Técnica (Tareas 3, 7, 10):**
- [ ] **Explicación Teórica:**
  - Cálculo de `xCoord` y `yCoord`
  - Justificación del centrado de origen
  - Explicación de inversión del eje Y
- [ ] **Análisis Memoria Constante:**
  - Párrafo explicando implementación
  - Efecto en rendimiento vs memoria global
  - Diagrama de flujo de datos
- [ ] **Análisis Memoria Compartida:**
  - Párrafo explicando implementación
  - Efecto en rendimiento vs otras versiones
  - Diagrama de flujo de datos

### **Informe Final:**
- [ ] Documento PDF con formato UVG
- [ ] Bitácoras de tiempo consolidadas
- [ ] Análisis comparativo de las 3 implementaciones
- [ ] Conclusiones sobre uso de diferentes tipos de memoria

---

## 📊 ENTREGABLES FINALES

### Código (45 puntos)
- [x] Versión CUDA funcional con 3 tipos de memoria
- [x] Generación de imagen con líneas detectadas
- [x] Documentación y comentarios
- [x] Uso correcto de barreras (`__syncthreads()`)
- [x] Liberación completa de memoria

### Informe PDF (20 puntos)
- [ ] Mínimo 1 página sobre algoritmo + implementación CUDA
- [ ] Formato UVG: carátula, índice, introducción, cuerpo, conclusiones
- [ ] Bitácoras de tiempo (mínimo 10 mediciones × 3 versiones)
- [ ] Análisis memoria Constante + diagrama
- [ ] Análisis memoria Compartida + diagrama
- [ ] Mínimo 3 citas bibliográficas

### Presentación (20 puntos)
- [ ] Presentación ejecutiva del proyecto
- [ ] Vestimenta business casual
- [ ] Calificación individual según participación

### Repositorio
- [ ] Código subido (no solo link)
- [ ] Link al repositorio

---

## 🔧 CONSIDERACIONES TÉCNICAS

### Memorias CUDA

| Tipo | Ubicación | Scope | Velocidad | Uso en Proyecto |
|------|-----------|-------|-----------|-----------------|
| **Global** | Device DRAM | Todos los threads | Lenta (~400 ciclos) | Imagen input, acumulador final |
| **Constante** | Device + Cache | Read-only, todos | Rápida si broadcast | Tablas sin/cos (90 valores) |
| **Compartida** | On-chip SM | Por bloque | Muy rápida (~4 ciclos) | Acumulador local por bloque |

### Límites Importantes
- Memoria Constante: 64 KB total, 8 KB cache por SM
- Memoria Compartida: depende del GPU (~48 KB por SM típico)
- `localAcc` requiere: `90 bins × 100 bins × 4 bytes = 36 KB` ✅ cabe

### Operaciones Críticas
- `atomicAdd`: necesario para evitar race conditions
- `__syncthreads()`: sincronizar threads de un bloque
- `__fmaf_rn()`: multiply-add optimizado
- `__float2int_rn()`: conversión con redondeo

---

## 🎨 DETALLES DE IMPLEMENTACIÓN

### Sistema de Coordenadas
```
Imagen original (0,0) = esquina superior izquierda

Transformado a:
      x
      ↑
      |
←-----+----→ y
      |
      ↓
      
Centro = (w/2, h/2)
xCoord = i - xCent
yCoord = yCent - j  (invertido!)
```

### Flujo del Algoritmo
1. Leer imagen PGM (blanco y negro)
2. Pre-calcular sin/cos en host
3. Copiar imagen y tablas a GPU
4. **Cada thread procesa 1 pixel:**
   - Si pixel > 0: votar por 90 líneas posibles
   - Cada voto incrementa `acc[rIdx][tIdx]`
5. Copiar acumulador a host
6. Encontrar bins con más votos = líneas detectadas
7. Dibujar líneas sobre imagen original

---

## 📚 RECURSOS Y REFERENCIAS

### Documentación Oficial
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Performance Metrics](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)

### Conceptos Clave a Investigar
- Hough Transform
- Constant Memory caching y broadcasting
- Shared Memory bank conflicts
- Atomic operations en CUDA
- Thread synchronization

---

## ⚠️ NOTAS IMPORTANTES

1. **No usar `cudaMalloc` para memoria Constante**, solo declarar con `__constant__`
2. **Usar `cudaMemcpyToSymbol`** en lugar de `cudaMemcpy` para constante
3. **Siempre sincronizar** antes y después de usar shared memory
4. **Threshold para líneas:** experimentar con valores (ej: max/2, promedio+2σ)
5. **Formato imagen salida:** cualquier formato común (PNG, JPG)
6. **Librería para dibujar:** puede usar OpenCV, STB, o similar

---

---

## 🚀 INSTRUCCIONES PARA CONTINUACIÓN

### **Compilación:**
```bash
make
```

### **Ejecución:**
```bash
# Ejecutar todas las versiones (recomendado para benchmarking)
./hough imagen.pgm

# Ejecutar versión específica
./hough imagen.pgm 1  # Solo Global Memory
./hough imagen.pgm 2  # Solo Constant Memory
./hough imagen.pgm 3  # Solo Shared Memory
```

### **Archivos de Salida:**
- `output_v1.pgm`, `output_v2.pgm`, `output_v3.pgm` - Acumuladores
- `lines_v1.ppm`, `lines_v2.ppm`, `lines_v3.ppm` - Líneas detectadas

---

**Estado Actual del Código:**
✅ **TODAS las implementaciones de código COMPLETADAS**
✅ 3 versiones funcionales: Global, Constante, Compartida
✅ Visualización de líneas implementada
✅ Código modular para benchmarking
⏳ **Pendiente:** Solo benchmarking y documentación