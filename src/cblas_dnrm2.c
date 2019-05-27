//
//  cblas_dnrm2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

double cblas_dnrm2(int n, double *x, int incx)
{
    const double one=1.0;
    const double zero=0.0;
    double norm=0.0;
    if(n==1)
    {
        norm=fabs(x[0]);
    }
    else if(n>1)
    {
        double scale=zero;
        double ssq=one;
        for(int i=0;i<n*incx;i+=incx)
        {
            if(x[i]!=zero)
            {
                double a=fabs(x[i]);
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
    }
    return norm;
}
