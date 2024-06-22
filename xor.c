#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

float or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

float xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float and[] = {
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};

float nand[] = {
    0, 0, 1,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};


int main(void)
{
    // srand(time(0));
    srand(69);

    float *td = and;
    
    size_t stride = 3;
    size_t n = 4;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    float rate = 1e-1;
    
    printf("cost = %f\n", nn_cost(nn, ti, to));
    for (size_t i = 0; i < 1; i++) {
#if 1
        float eps = 1e-1;
        nn_finite_diff(nn, g, eps, ti, to);
#else
        nn_backprop(nn, g, ti, to);
#endif            
        NN_PRINT(g);
        // nn_learn(nn, g, rate);
        // `printf("%zu: cost = %f\n", i, nn_cost(nn, ti, to));
    }
    
    for (size_t i = 0; i < 2; i ++) {
        for (size_t j = 0; j < 2; j ++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));       
        }
    }    
    return 0;
}
