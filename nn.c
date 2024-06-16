#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

int main(void)
{
  srand(time(0));

  Mat x = mat_alloc(1, 2);
  
  Mat w1 = mat_alloc(1, 2);
  Mat b1 = mat_alloc(2, 1);
  Mat a1 = mat_alloc(1, 2);
  
  Mat w2 = mat_alloc(1, 2);
  Mat b2 = mat_alloc(1, 1);  

  mat_rand(x, 0, 1);
  mat_rand(w1, 0, 1);
  mat_rand(b1, 0, 1);
  mat_rand(a1, 0, 1);
  mat_rand(w2, 0, 1);
  mat_rand(b2, 0, 1);

  MAT_AT(x, 0, 0) = 0;
  MAT_AT(x, 0, 1) = 1;

  // forward pass - signle layer
  mat_dot(a1, x, w1);
  mat_sum(a1, b1);
  mat_sig(a1);
  
  MAT_PRINT(w1);
  MAT_PRINT(b1);
  MAT_PRINT(w2);
  MAT_PRINT(b2);
  
  return 0;
}