//
//  cblas_dsdot.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

double cblas_dsdot(int n, float *x, int incx, float *y, int incy)
{
    double sum=0.0;
    if((incx==1)&&(incy==1))
        for(int i=0;i<n;i++)
            sum+=((double)x[i])*((double)y[i]);
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
            sum+=((double)x[i*incx])*((double)y[i*incy]);
    }
    return sum;
}
