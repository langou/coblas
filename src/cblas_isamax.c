//
//  cblas_isamax.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

int cblas_isamax(int n, float  *x, int incx)
{
    int imax=0;
    if((n==1)&&(incx>0))
        imax=0;
    else if((n>1)&&(incx==1))
    {
        float maxval=fabsf(x[0]);
        imax=0;
        for(int i=1;i<n;i++)
        {
            if(fabsf(x[i])>maxval)
            {
                imax=i;
                maxval=fabsf(x[i]);
            }
        }
    }
    else if((n>1)&&(incx>1))
    {
        float maxval=fabsf(x[0]);
        imax=0;
        int ix=incx;
        for(int i=1;i<n;i++)
        {
            if(fabsf(x[ix])>maxval)
            {
                imax=i;
                maxval=fabsf(x[ix]);
            }
            ix+=incx;
        }
    }
    return imax;
}
