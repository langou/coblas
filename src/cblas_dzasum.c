//
//  cblas_dzasum.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

double cblas_dzasum(int n, double complex *x, int incx)
{
    double sum=0.0;
    if(incx==1)
        for(int i=0;i<n;i++)
            sum+=fabs(creal(x[i]))+fabs(cimag(x[i]));
    else if(incx>1)
        for(int i=0;i<n*incx;i+=incx)
            sum+=fabs(creal(x[i]))+fabs(cimag(x[i]));
    return sum;
}
