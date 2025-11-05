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
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

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

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  if (argc < 2) {
    fprintf(stderr, "Uso: %s input.pgm [output_accum.pgm]\n", argv[0]);
    return 1;
  }
  
  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);


  // Tabla de seno/coseno en host (mismo muestreo que degreeBins)
  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  computeSinCosTable(pcCos, pcSin, degreeBins);

  float rMax   = sqrtf((float)w * w + (float)h * h) / 2.0f;
  float rScale = 2.0f * rMax / rBins;

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht, pcCos, pcSin, rMax, rScale);

  printf("Params: w=%d h=%d degreeBins=%d rBins=%d rMax=%.9f rScale=%.9f\n",
       w, h, degreeBins, rBins, rMax, rScale);

  // Copiar LUT sin/cos (host → device global). Versión baseline sin optimizar.
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = (w * h + 256 - 1) / 256;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  accumulateHoughGPU <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  // Chequeo de error de lanzamiento del kernel
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Kernel (baseline global) time: %.3f ms\n", ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA runtime error after sync: %s\n", cudaGetErrorString(err));
  }

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  saveAccumulatorAsPGM("output_gpu.pgm", h_hough, rBins, degreeBins);
  printf("Acumulador guardado en output_gpu.pgm (alto=rBins=%d, ancho=degreeBins=%d)\n",
        rBins, degreeBins);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  // clean-up

  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  free(h_hough);
  delete[] cpuht;   // cpuht fue creado con new[] en CPU_HoughTran
  free(pcCos);
  free(pcSin);

  return 0;
}
