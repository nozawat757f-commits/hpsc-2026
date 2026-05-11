#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < range) bucket[i] = 0;
}

__global__ void add_bucket(int *key, int *bucket, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bucket[key[i]], 1);
}

int main() {
    int n = 50;
    int range = 5;
    int *key, *bucket;

    cudaMallocManaged(&key, n * sizeof(int));
    cudaMallocManaged(&bucket, range * sizeof(int));

    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    int blockSize = 32;
    init_bucket<<<(range + blockSize - 1) / blockSize, blockSize>>>(bucket, range);
    cudaDeviceSynchronize();

    add_bucket<<<(n + blockSize - 1) / blockSize, blockSize>>>(key, bucket, n);
    cudaDeviceSynchronize();

    for (int i=0, j=0; i<range; i++) {
        for (; bucket[i]>0; bucket[i]--) {
            key[j++] = i;
        }
    }

    for (int i=0; i<n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(bucket);

    return 0;
}