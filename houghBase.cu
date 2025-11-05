/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
// const float radInc = degreeInc * M_PI / 180;

// HELPERS
static inline int rnint_to_int_host(float v) {
  // nearest-even en CPU
  return (int)nearbyintf(v);
}

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc,
                   const float *pcCos, const float *pcSin,
                   float rMax, float rScale)
{
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, sizeof(int) * rBins * degreeBins);

  int xCent = w / 2;
  int yCent = h / 2;

  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < h; ++j) {
      int idx = j * w + i;
      if (pic[idx] > 0) {
        int xCoord = i - xCent;
        int yCoord = yCent - j;  // igual que en GPU

        for (int tIdx = 0; tIdx < degreeBins; ++tIdx) {
          float r = fmaf((float)xCoord, pcCos[tIdx], (float)yCoord * pcSin[tIdx]);
          float v = (r + rMax) / rScale;
          int rIdx = rnint_to_int_host(v);

          // clamp idéntico al GPU
          if (rIdx < 0) rIdx = 0;
          else if (rIdx >= rBins) rIdx = rBins - 1;

          (*acc)[rIdx * degreeBins + tIdx]++; // en CPU es ++
        }
      }
    }
  }
}

//*****************************************************************
// Memoria constante para tablas de senos y cosenos
__constant__ float d_Cos_const[degreeBins];
__constant__ float d_Sin_const[degreeBins];

//*****************************************************************
// GPU kernel con memoria Compartida: acumulador local por bloque
__global__ void GPU_HoughTranShared(const unsigned char *pic, int w, int h,
                                    int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;

  // Acumulador local en shared memory
  __shared__ int localAcc[degreeBins * rBins];

  // Inicializar acumulador local a 0 (distribuir entre threads del bloque)
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
    localAcc[i] = 0;
  }

  // Barrera de sincronización
  __syncthreads();

  // Solo procesar si el thread tiene un pixel válido
  if (gloID < w * h) {
    const int xCent = w / 2;
    const int yCent = h / 2;

    // Origen al centro y eje Y hacia arriba
    int xCoord = (gloID % w) - xCent;
    int yCoord = yCent - (gloID / w);

    // Solo votan los píxeles > 0
    if (pic[gloID] > 0) {
      for (int tIdx = 0; tIdx < degreeBins; ++tIdx)
      {
        // Usar memoria constante para las tablas
        float r = __fmaf_rn((float)xCoord, d_Cos_const[tIdx], (float)yCoord * d_Sin_const[tIdx]);
        float v = (r + rMax) / rScale;
        int rIdx = __float2int_rn(v);
        if (rIdx < 0) rIdx = 0;
        else if (rIdx >= rBins) rIdx = rBins - 1;

        // Votar en acumulador LOCAL con atomicAdd
        atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
      }
    }
  }

  // Segunda barrera de sincronización
  __syncthreads();

  // Copiar acumulador local → global (distribuir entre threads del bloque)
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
    if (localAcc[i] > 0) {
      atomicAdd(&acc[i], localAcc[i]);
    }
  }
}
// GPU kernel con memoria Constante: usa tablas cos/sin en memoria constante
__global__ void GPU_HoughTranConst(const unsigned char *pic, int w, int h,
                                   int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  const int xCent = w / 2;
  const int yCent = h / 2;

  // Origen al centro y eje Y hacia arriba
  int xCoord = (gloID % w) - xCent;
  int yCoord = yCent - (gloID / w);

  // Sólo votan los píxeles > 0
  if (pic[gloID] == 0) return;

  for (int tIdx = 0; tIdx < degreeBins; ++tIdx)
  {
    // Usar memoria constante directamente
    float r = __fmaf_rn((float)xCoord, d_Cos_const[tIdx], (float)yCoord * d_Sin_const[tIdx]);
    float v = (r + rMax) / rScale;
    int rIdx = __float2int_rn(v);
    if (rIdx < 0) rIdx = 0;
    else if (rIdx >= rBins) rIdx = rBins - 1;

    atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
  }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
// GPU kernel baseline: 1 hilo por píxel, usa tablas cos/sin en global
__global__ void accumulateHoughGPU(const unsigned char *pic, int w, int h,
                                   int *acc, float rMax, float rScale,
                                   const float *d_Cos, const float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  const int xCent = w / 2;
  const int yCent = h / 2;

  // Origen al centro y eje Y hacia arriba
  int xCoord = (gloID % w) - xCent;
  int yCoord = yCent - (gloID / w);

  // Sólo votan los píxeles > 0
  if (pic[gloID] == 0) return;

  for (int tIdx = 0; tIdx < degreeBins; ++tIdx)
  {
    float r = __fmaf_rn((float)xCoord, d_Cos[tIdx], (float)yCoord * d_Sin[tIdx]);
    float v = (r + rMax) / rScale;
    int rIdx = __float2int_rn(v);
    if (rIdx < 0) rIdx = 0;
    else if (rIdx >= rBins) rIdx = rBins - 1;

    atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
  }
}

//*****************************************************************
// Helper: precompute sin/cos table on host
static void computeSinCosTable(float *pcCos, float *pcSin, int degreeBins) {
  const float deg2rad = (float)M_PI / 180.0f;
  const float stepDeg = 180.0f / degreeBins; // con degreeInc=2 → 2°
  for (int t = 0; t < degreeBins; ++t) {
    float theta = (t * stepDeg) * deg2rad;
    pcCos[t] = cosf(theta);
    pcSin[t] = sinf(theta);
  }
}

// Guarda el acumulador (r x theta) como PGM P5 normalizado 0..255
static void saveAccumulatorAsPGM(const char *path, const int *h_acc,
                                int rBins, int degreeBins)
{
  const int total = rBins * degreeBins;
  int vmax = 0;
  for (int i = 0; i < total; ++i) if (h_acc[i] > vmax) vmax = h_acc[i];
  if (vmax == 0) vmax = 1;

  unsigned char *img = (unsigned char *)malloc(total);
  for (int i = 0; i < total; ++i) {
    float v = (float)h_acc[i] / (float)vmax;
    int gi = (int)lrintf(255.0f * v);
    if (gi < 0) gi = 0; if (gi > 255) gi = 255;
    img[i] = (unsigned char)gi;
  }

  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "No pude abrir %s para escribir.\n", path); free(img); return; }
  // Nota: ancho = degreeBins, alto = rBins (imagen del acumulador)
  fprintf(f, "P5\n%d %d\n255\n", degreeBins, rBins);
  fwrite(img, 1, total, f);
  fclose(f);
  free(img);
}

// Detecta líneas con votos > threshold y las dibuja sobre la imagen original
static void drawDetectedLines(const char *outputPath, const unsigned char *originalImg,
                             int w, int h, const int *h_acc, int rBins, int degreeBins,
                             const float *pcCos, const float *pcSin, float rMax, float rScale)
{
  // Calcular threshold: promedio + 2*stddev o max/4, lo que sea mayor
  int total = rBins * degreeBins;
  int sum = 0, vmax = 0;
  for (int i = 0; i < total; i++) {
    sum += h_acc[i];
    if (h_acc[i] > vmax) vmax = h_acc[i];
  }
  float mean = (float)sum / total;

  float variance = 0.0f;
  for (int i = 0; i < total; i++) {
    float diff = h_acc[i] - mean;
    variance += diff * diff;
  }
  float stddev = sqrtf(variance / total);

  int threshold = (int)(mean + 2.0f * stddev);
  int threshold_alt = vmax / 4;
  if (threshold_alt > threshold) threshold = threshold_alt;

  printf("Threshold para detección de líneas: %d (mean=%.1f, stddev=%.1f, max=%d)\n",
         threshold, mean, stddev, vmax);

  // Crear imagen RGB (3 canales) para dibujar líneas a color
  unsigned char *rgbImg = (unsigned char *)malloc(w * h * 3);
  for (int i = 0; i < w * h; i++) {
    // Convertir escala de grises a RGB
    rgbImg[i*3] = originalImg[i];     // R
    rgbImg[i*3+1] = originalImg[i];   // G
    rgbImg[i*3+2] = originalImg[i];   // B
  }

  int linesDrawn = 0;

  // Buscar picos en el acumulador y dibujar líneas
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int votes = h_acc[rIdx * degreeBins + tIdx];

      if (votes > threshold) {
        // Convertir de índices a parámetros reales
        float r = (rIdx * rScale) - rMax;
        float cosTheta = pcCos[tIdx];
        float sinTheta = pcSin[tIdx];

        // Dibujar la línea r = x*cos(θ) + y*sin(θ)
        // Para cada x, calcular y = (r - x*cos(θ)) / sin(θ)
        // Para cada y, calcular x = (r - y*sin(θ)) / cos(θ)

        for (int x = 0; x < w; x++) {
          if (fabs(sinTheta) > 0.001f) { // Evitar división por cero
            float y_real = (r - (x - w/2) * cosTheta) / sinTheta + h/2;
            int y = (int)lrintf(y_real);
            if (y >= 0 && y < h) {
              int idx = y * w + x;
              // Dibujar en rojo
              rgbImg[idx*3] = 255;     // R
              rgbImg[idx*3+1] = 0;     // G
              rgbImg[idx*3+2] = 0;     // B
            }
          }
        }

        for (int y = 0; y < h; y++) {
          if (fabs(cosTheta) > 0.001f) { // Evitar división por cero
            float x_real = (r - (y - h/2) * sinTheta) / cosTheta + w/2;
            int x = (int)lrintf(x_real);
            if (x >= 0 && x < w) {
              int idx = y * w + x;
              // Dibujar en rojo
              rgbImg[idx*3] = 255;     // R
              rgbImg[idx*3+1] = 0;     // G
              rgbImg[idx*3+2] = 0;     // B
            }
          }
        }

        linesDrawn++;
        printf("Línea detectada: r=%.2f, θ=%d°, votos=%d\n",
               r, tIdx * 2, votes);
      }
    }
  }

  printf("Total de líneas dibujadas: %d\n", linesDrawn);

  // Guardar como PPM P6 (color)
  FILE *f = fopen(outputPath, "wb");
  if (!f) {
    fprintf(stderr, "No pude abrir %s para escribir.\n", outputPath);
    free(rgbImg);
    return;
  }

  fprintf(f, "P6\n%d %d\n255\n", w, h);
  fwrite(rgbImg, 3, w * h, f);
  fclose(f);
  free(rgbImg);

  printf("Imagen con líneas guardada en: %s\n", outputPath);
}

//*****************************************************************
// Función auxiliar para ejecutar una versión específica del kernel
static float runKernelVersion(int version, unsigned char *d_in, int w, int h,
                             int *d_hough, float rMax, float rScale,
                             float *d_Cos, float *d_Sin)
{
  int blockNum = (w * h + 256 - 1) / 256;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  switch(version) {
    case 1: // Global memory
      accumulateHoughGPU<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
      break;
    case 2: // Constant memory
      GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);
      break;
    case 3: // Shared memory
      GPU_HoughTranShared<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);
      break;
    default:
      fprintf(stderr, "Versión de kernel inválida: %d\n", version);
      return -1.0f;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    return -1.0f;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ms;
}

//*****************************************************************
int main (int argc, char **argv)
{
  

  printf("=== LIMPIEZA ===\n");

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  free(h_hough);
  delete[] cpuht;
  free(pcCos);
  free(pcSin);

  printf("Ejecución completada\n");
  return 0;
}