//
//  cblas_dznrm2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

double cblas_dznrm2(int n, double complex *x, int incx)
{
    const double one=1.0;
    const double zero=0.0;
    double norm=0.0;
    double scale=zero;
    double ssq=one;
    for(int i=0;i<n*incx;i+=incx)
    {
        if(creal(x[i])!=zero)
        {
            double a=fabs(creal(x[i]));
            if(scale<a)
            {
                double b=scale/a;
                ssq=one+ssq*b*b;
                scale=a;
            }
            else
            {
                double b=a/scale;
                ssq+=b*b;
            }
        }
        if(cimag(x[i])!=zero)
        {
            double a=fabs(cimag(x[i]));
            if(scale<a)
            {
                double b=scale/a;
                ssq=one+ssq*b*b;
                scale=a;
            }
            else
            {
                double b=a/scale;
                ssq+=b*b;
            }
        }
    }
    norm=scale*sqrt(ssq);
    return norm;
}
