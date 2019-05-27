//
//  cblas_scasum.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

float cblas_scasum(int n, float complex *x, int incx)
{
    float sum=0.0f;
    if(incx==1)
        for(int i=0;i<n;i++)
            sum+=fabsf(crealf(x[i]))+fabsf(cimagf(x[i]));
    else if(incx>1)
        for(int i=0;i<n*incx;i+=incx)
            sum+=fabsf(crealf(x[i]))+fabsf(cimagf(x[i]));
    return sum;
}
