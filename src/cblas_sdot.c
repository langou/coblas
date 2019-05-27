//
//  cblas_sdot.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

float cblas_sdot(int n, float  *x, int incx, float  *y, int incy)
{
    float sum=0.0f;
    if((incx==1)&&(incy==1))
        for(int i=0;i<n;i++)
            sum+=x[i]*y[i];
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
            sum+=x[i*incx]*y[i*incy];
    }
    return sum;
}
