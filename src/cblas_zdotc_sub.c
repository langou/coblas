//
//  cblas_zdotc_sub.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zdotc_sub(int n, double complex *x, int incx, double complex *y, int incy, double complex *ret)
{
    double complex sum=0.0;
    if((incx==1)&&(incy==1))
        for(int i=0;i<n;i++)
            sum+=conj(x[i])*y[i];
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
            sum+=conj(x[i*incx])*y[i*incy];
    }
    *ret=sum;
}
