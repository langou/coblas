//
//  cblas_sasum.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

float cblas_sasum(int n, float *x, int incx)
{
    float sum=0.0f;
    if(incx==1)
        for(int i=0;i<n;i++)
            sum+=fabsf(x[i]);
    else if(incx>1)
        for(int i=0;i<n*incx;i+=incx)
            sum+=fabsf(x[i]);
    return sum;
}
