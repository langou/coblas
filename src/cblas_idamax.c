//
//  cblas_idamax.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

int cblas_idamax(int n, double *x, int incx)
{
    int imax=0;
    if((n==1)&&(incx>0))
        imax=0;
    else if((n>1)&&(incx==1))
    {
        double maxval=fabs(x[0]);
        imax=0;
        for(int i=1;i<n;i++)
        {
            if(fabs(x[i])>maxval)
            {
                imax=i;
                maxval=fabs(x[i]);
            }
        }
    }
    else if((n>1)&&(incx>1))
    {
        double maxval=fabs(x[0]);
        imax=0;
        int ix=incx;
        for(int i=1;i<n;i++)
        {
            if(fabs(x[ix])>maxval)
            {
                imax=i;
                maxval=fabs(x[ix]);
            }
            ix+=incx;
        }
    }
    return imax;
}
