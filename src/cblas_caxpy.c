//
//  cblas_caxpy.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_caxpy(int n, float complex *alpha, float complex *x, int incx, float complex *y, int incy)
{
    const float complex a=(*alpha);
    if((incx==1)&&(incy==1))
        for(int i=0;i<n;i++)
            y[i]+=a*x[i];
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
            y[i*incy]+=a*x[i*incx];
    }
}
