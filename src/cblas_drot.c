//
//  cblas_drot.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_drot(int n, double *x, int incx, double *y, int incy, double c, double  s)
{
    if((incx==1)&&(incy==1))
    {
        for(int i=0;i<n;i++)
        {
            double temp=c*x[i]+s*y[i];
            y[i]=c*y[i]-s*x[i];
            x[i]=temp;
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
            double temp=c*x[i*incx]+s*y[i*incy];
            y[i*incy]=c*y[i*incy]-s*x[i*incx];
            x[i*incx]=temp;
        }
    }
}
