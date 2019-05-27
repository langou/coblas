//
//  cblas_cdotu_sub.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_cdotu_sub(int n, float complex *x, int incx, float complex *y, int incy, float complex *ret)
{
    float complex sum=0.0f;
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
    *ret=sum;
}
