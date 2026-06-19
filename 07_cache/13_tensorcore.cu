#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>

using namespace std;
using namespace nvcuda;

// Gemini Optimized v6:
// 1. Vectorized Global Memory Loads (float4) to maximize bandwidth.
// 2. Shared Memory Padding (+8) to eliminate bank conflicts.
// 3. True Double Buffering for latency hiding.

#define BM 256
#define BN 128
#define BK 16

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

  // Padding (+8) to avoid shared memory bank conflicts during load_matrix_sync
  __shared__ half sh_a[2][BK][BM + 8];
  __shared__ half sh_b[2][BK][BN + 8];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WFRAG_M][WFRAG_N];
  for (int r = 0; r < WFRAG_M; r++)
    for (int c = 0; c < WFRAG_N; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int num_k_tiles = dim_k / BK;

  // Prologue: Load Stage 0 with float4 vectorized loads
  int k0 = 0;
  for (int i = tid; i < (BK * BM) / 4; i += NTHREADS) {
    int col = i / (BM / 4);
    int row4 = i % (BM / 4);
    int row = row4 * 4;
    float4 a_vec = reinterpret_cast<const float4*>(&d_a[(k0 + col) * dim_m + block_m + row])[0];
    sh_a[0][col][row + 0] = __float2half(a_vec.x);
    sh_a[0][col][row + 1] = __float2half(a_vec.y);
    sh_a[0][col][row + 2] = __float2half(a_vec.z);
    sh_a[0][col][row + 3] = __float2half(a_vec.w);
  }
  for (int i = tid; i < (BN * BK) / 4; i += NTHREADS) {
    int col = i / (BK / 4);
    int row4 = i % (BK / 4);
    int row = row4 * 4;
    float4 b_vec = reinterpret_cast<const float4*>(&d_b[(block_n + col) * dim_k + k0 + row])[0];
    sh_b[0][row + 0][col] = __float2half(b_vec.x);
    sh_b[0][row + 1][col] = __float2half(b_vec.y);
    sh_b[0][row + 2][col] = __float2half(b_vec.z);
    sh_b[0][row + 3][col] = __float2half(b_vec.w);
  }
  __syncthreads();

  for (int t = 0; t < num_k_tiles; t++) {
    int cur = t % 2;
    int nxt = (t + 1) % 2;

    // Prefetch next tile while computing current
    if (t + 1 < num_k_tiles) {
      int k_next = (t + 1) * BK;
      for (int i = tid; i < (BK * BM) / 4; i += NTHREADS) {
        int col = i / (BM / 4);
        int row4 = i % (BM / 4);
        int row = row4 * 4;
        float4 a_vec = reinterpret_cast<const float4*>(&d_a[(k_next + col) * dim_m + block_m + row])[0];
        sh_a[nxt][col][row + 0] = __float2half(a_vec.x);
        sh_a[nxt][col][row + 1] = __float2half(a_vec.y);
        sh_a[nxt][col][row + 2] = __float2half(a_vec.z);
        sh_a[nxt][col][row + 3] = __float2half(a_vec.w);
      }
      for (int i = tid; i < (BN * BK) / 4; i += NTHREADS) {
        int col = i / (BK / 4);
        int row4 = i % (BK / 4);
        int row = row4 * 4;
        float4 b_vec = reinterpret_cast<const float4*>(&d_b[(block_n + col) * dim_k + k_next + row])[0];
        sh_b[nxt][row + 0][col] = __float2half(b_vec.x);
        sh_b[nxt][row + 1][col] = __float2half(b_vec.y);
        sh_b[nxt][row + 2][col] = __float2half(b_vec.z);
        sh_b[nxt][row + 3][col] = __float2half(b_vec.w);
      }
    }

    // Compute on current stage
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[WFRAG_M];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WFRAG_N];
      
      for (int r = 0; r < WFRAG_M; r++) {
        int m_off = warp_m * WM + r * 16;
        wmma::load_matrix_sync(a_frag[r], &sh_a[cur][kk][m_off], BM + 8); // Padded stride
      }
      for (int c = 0; c < WFRAG_N; c++) {
        int n_off = warp_n * WN + c * 16;
        wmma::load_matrix_sync(b_frag[c], &sh_b[cur][kk][n_off], BN + 8); // Padded stride
      }
      for (int r = 0; r < WFRAG_M; r++)
        for (int c = 0; c < WFRAG_N; c++)
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }
    __syncthreads();
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
