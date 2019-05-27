//
//  cblas_sdsdot.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

float cblas_sdsdot(int n, float alpha, float *x, int incx, float *y, int incy)
{
    double sum=(double)(alpha);
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
    float sumf=(float)sum;
    return sumf;
}
