#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 x_vec = _mm512_load_ps(x);
  __m512 y_vec = _mm512_load_ps(y);
  __m512 m_vec = _mm512_load_ps(m);

  __m512i j_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __mmask16 mask = _mm512_cmp_epi32_mask(j_idx, _mm512_set1_epi32(i), _MM_CMPINT_NE);
    
    __m512 rx = _mm512_sub_ps(xi, x_vec);
    __m512 ry = _mm512_sub_ps(yi, y_vec);

    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    
    __m512 rinv = _mm512_rsqrt14_ps(r2);
    
    __m512 rinv3 = _mm512_mul_ps(_mm512_mul_ps(rinv, rinv), rinv);


    __m512 fxi_v = _mm512_mul_ps(_mm512_mul_ps(rx, m_vec), rinv3);
    __m512 fyi_v = _mm512_mul_ps(_mm512_mul_ps(ry, m_vec), rinv3);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, fxi_v);
    fy[i] -= _mm512_mask_reduce_add_ps(mask, fyi_v);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
  
  return 0;
}
