#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

// Larger warp tile: each warp computes WM x WN via more fragments,
// reducing the number of warps needed and increasing arithmetic intensity.
// 256x128 block tile, 256 threads (8 warps), arranged 4(m) x 2(n)
#define BM 256
#define BN 128
#define BK 32

#define WARPS_M 4
#define WARPS_N 2
#define WM (BM / WARPS_M)   // 64
#define WN (BN / WARPS_N)   // 64
#define WFRAG_M (WM / 16)   // 4
#define WFRAG_N (WN / 16)   // 4

#define NTHREADS (WARPS_M * WARPS_N * 32)  // 256

__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       float *d_a, float *d_b, float *d_c) {
  int block_m = BM * blockIdx.x;
  int block_n = BN * blockIdx.y;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int warp_m = warp_id / WARPS_N;
  int warp_n = warp_id % WARPS_N;

  __shared__ half sh_a[BK][BM];
  __shared__ half sh_b[BK][BN];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WFRAG_M][WFRAG_N];
  for (int r = 0; r < WFRAG_M; r++)
    for (int c = 0; c < WFRAG_N; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k0 = 0; k0 < dim_k; k0 += BK) {
    __syncthreads();
    for (int idx = tid; idx < BK * BM; idx += NTHREADS) {
      int kk = idx / BM;
      int mm = idx % BM;
      sh_a[kk][mm] = __float2half(d_a[(k0 + kk) * dim_m + block_m + mm]);
    }
    for (int idx = tid; idx < BK * BN; idx += NTHREADS) {
      int kk = idx / BN;
      int nn = idx % BN;
      sh_b[kk][nn] = __float2half(d_b[(block_n + nn) * dim_k + k0 + kk]);
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[WFRAG_M];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WFRAG_N];
      for (int r = 0; r < WFRAG_M; r++) {
        int m_off = warp_m * WM + r * 16;
        wmma::load_matrix_sync(a_frag[r], &sh_a[kk][m_off], BM);
      }
      for (int c = 0; c < WFRAG_N; c++) {
        int n_off = warp_n * WN + c * 16;
        wmma::load_matrix_sync(b_frag[c], &sh_b[kk][n_off], BN);
      }
      for (int r = 0; r < WFRAG_M; r++)
        for (int c = 0; c < WFRAG_N; c++)
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }
  }

  for (int r = 0; r < WFRAG_M; r++) {
    for (int c = 0; c < WFRAG_N; c++) {
      int c_m = block_m + warp_m * WM + r * 16;
      int c_n = block_n + warp_n * WN + c * 16;
      if (c_m < dim_m && c_n < dim_n)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m,
                                wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k,
                 &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k))
                    + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block = dim3(NTHREADS);
  dim3 grid = dim3((m + BM - 1) / BM, (n + BN - 1) / BN);

  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<<grid, block>>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tkernel = chrono::duration<double>(toc - tic).count() / Nt;
  double kernel_flops = double(num_flops) / tkernel / 1.0e9;

  printf("CUBLAS: %.2f Gflops, TensorCore: %.2f Gflops\n",
         cublas_flops, kernel_flops);

  double err = 0;
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      err += fabs(C[m*i+j] - C2[m*i+j]);
  printf("error: %lf\n", err/n/m);

  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
  cublasDestroy(cublas_handle);
}
