#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits.h>

void printbits(uint64_t n) {
    uint64_t i; 
    i = 1UL<<(sizeof(n)*CHAR_BIT-1);
    printf("Printing bits for %llu:\n", n);
    while(i>0){
         if(n&i)
              printf("1"); 
         else 
              printf("0"); 
         i >>= 1;
    }
    printf("\n");
}

uint32_t* weyl_mid_sq(uint64_t *a, uint64_t *b, size_t n)
{
    uint64_t x, y, z;
    uint32_t* ret = (uint32_t*) malloc(n * sizeof(uint32_t));

    for (size_t i = 0; i < n; i++)
    {
        //printf("%llu %llu\n", a[i], b[i]);

        y = x = a[i] * b[i];
        z = y + b[i];

        // Round 1
        x = x*x + y;
        //printbits(x);
        //printbits(x>>32);
        //printbits(x<<32);
        x = (x>>32) | (x<<32);
        //printbits(x);
        //printf("After round 1: %llu\n", x);

        // Round 2
        x = x*x + z;
        x = (x>>32) | (x<<32);
        //printf("After round 2: %llu\n", x);

        // Round 3
        x = x*x + y;
        x = (x>>32) | (x<<32);
        //printf("After round 3: %llu\n", x);

        // Round 4
        ret[i] = (x*x + z) >> 32;
        //printf("After round 4: %llu\n", ret[i]);
    }
    return ret;
}

uint32_t* midsq(uint64_t *a, uint64_t *b, size_t n)
{
    uint64_t x;
    uint32_t* ret = (uint32_t*) malloc(n * sizeof(uint32_t));

    for (size_t i = 0; i < n; i++)
    {
        // Multiply
        x = a[i] * b[i];

        // Shift
        x = x>>32;

        ret[i] = x;
    }
    return ret;
}

void freeArray(uint32_t *b) {
    free(b);
}

