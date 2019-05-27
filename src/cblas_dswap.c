//
//  cblas_dswap.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dswap(int n, double *x, int incx, double *y, int incy)
{
    if((incx==1)&&(incy==1))
    {
        for(int i=0;i<n;i++)
        {
            double temp=x[i];
            x[i]=y[i];
            y[i]=temp;
        }
    }
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        for(int i=0;i<n;i++)
        {
            double temp=x[i*incx];
            x[i*incx]=y[i*incy];
            y[i*incy]=temp;
        }
    }
}
