//
//  cblas_zcopy.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zcopy(int n, double complex *x, int incx, double complex *y, int incy)
{
    if((incx==1)&&(incy==1))
        for(int i=0;i<n;i++)
            y[i]=x[i];
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
            y[i*incy]=x[i*incx];
    }
}
