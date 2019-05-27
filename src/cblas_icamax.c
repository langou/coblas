//
//  cblas_icamax.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

int cblas_icamax(int n, float complex *x, int incx)
{
    int imax=0;
    if((n==1)&&(incx>0))
        imax=0;
    else if((n>1)&&(incx==1))
    {
        float absval=fabsf(crealf(x[0]))+fabsf(cimagf(x[0]));
        float maxval=absval;
        imax=0;
        for(int i=1;i<n;i++)
        {
            absval=fabsf(crealf(x[i]))+fabsf(cimagf(x[i]));
            if(absval>maxval)
            {
                imax=i;
                maxval=absval;
            }
        }
    }
    else if((n>1)&&(incx>1))
    {
        float absval=fabsf(crealf(x[0]))+fabsf(cimagf(x[0]));
        float maxval=absval;
        imax=0;
        int ix=incx;
        for(int i=1;i<n;i++)
        {
            absval=fabsf(crealf(x[ix]))+fabsf(cimagf(x[ix]));
            if(absval>maxval)
            {
                imax=i;
                maxval=absval;
            }
            ix+=incx;
        }
    }
    return imax;
}
