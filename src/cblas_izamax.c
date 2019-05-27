//
//  cblas_izamax.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

int cblas_izamax(int n, double complex *x, int incx)
{
    int imax=0;
    if((n==1)&&(incx>0))
        imax=0;
    else if((n>1)&&(incx==1))
    {
        double absval=fabs(creal(x[0]))+fabs(cimag(x[0]));
        double maxval=absval;
        imax=0;
        for(int i=1;i<n;i++)
        {
            absval=fabs(creal(x[i]))+fabs(cimag(x[i]));
            if(absval>maxval)
            {
                imax=i;
                maxval=absval;
            }
        }
    }
    else if((n>1)&&(incx>1))
    {
        double absval=fabs(creal(x[0]))+fabs(cimag(x[0]));
        double maxval=absval;
        imax=0;
        int ix=incx;
        for(int i=1;i<n;i++)
        {
            absval=fabs(creal(x[ix]))+fabs(cimag(x[ix]));
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
