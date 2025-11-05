# Transformada de Hough en CUDA

Implementación paralela de la Transformada de Hough para detección de líneas en imágenes, usando tres tipos diferentes de memoria CUDA.

## 📋 Descripción

La Transformada de Hough es una técnica de Computer Vision para detectar líneas rectas en imágenes binarias. Este proyecto implementa el algoritmo en CUDA usando:

1. **Memoria Global** - Versión baseline
2. **Memoria Constante** - Optimización para tablas sin/cos
3. **Memoria Compartida** - Acumulador local por bloque

## 🚀 Compilación

```bash
make
```

## 📖 Uso

### Ejecutar todas las versiones (recomendado):
```bash
./hough imagen.pgm
```

### Ejecutar versión específica:
```bash
./hough imagen.pgm 1    # Solo Global Memory
./hough imagen.pgm 2    # Solo Constant Memory
./hough imagen.pgm 3    # Solo Shared Memory
```

## 📁 Archivos de Salida

Para cada versión ejecutada se generan:

- `output_v[1|2|3].pgm` - Acumulador de Hough (espacio paramétrico)
- `lines_v[1|2|3].ppm` - Imagen original con líneas detectadas en rojo

## ⚙️ Parámetros del Algoritmo

- **θ (ángulos):** 90 bins, incrementos de 2°, rango [0°, 180°)
- **r (distancias):** 100 bins, rango [-rMax, rMax]
- **Threshold:** `max(promedio + 2*desv_std, max/4)`

## 🔧 Implementación

### Versión 1: Global Memory
- Kernel baseline usando memoria global para tablas sin/cos
- 1 thread por pixel de la imagen
- Configuración: bloques de 256 threads

### Versión 2: Constant Memory
- Tablas sin/cos almacenadas en memoria constante
- Optimización para accesos broadcast
- Variables `__constant__` declaradas globalmente

### Versión 3: Shared Memory
- Acumulador local por bloque en memoria compartida
- Reducción de conflictos de memoria global
- Dos barreras `__syncthreads()` para sincronización

## 📊 Validación

Todas las versiones GPU se validan automáticamente contra la implementación CPU de referencia.

## 📚 Archivos del Proyecto

```
proyecto/
├── houghBase.cu         
├── common/pgm.h
├── common/pgm.cpp
├── Makefile
├── README.md
```