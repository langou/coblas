//
//  cblas_scnrm2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

float cblas_scnrm2(int n, float complex *x, int incx)
{
    const float one=1.0f;
    const float zero=0.0f;
    float norm=0.0f;
    float scale=zero;
    float ssq=one;
    for(int i=0;i<n*incx;i+=incx)
    {
        if(crealf(x[i])!=zero)
        {
            float a=fabsf(crealf(x[i]));
            if(scale<a)
            {
                float b=scale/a;
                ssq=one+ssq*b*b;
                scale=a;
            }
            else
            {
                float b=a/scale;
                ssq+=b*b;
            }
        }
        if(cimagf(x[i])!=zero)
        {
            float a=fabsf(cimagf(x[i]));
            if(scale<a)
            {
                float b=scale/a;
                ssq=one+ssq*b*b;
                scale=a;
            }
            else
            {
                float b=a/scale;
                ssq+=b*b;
            }
        }
    }
    norm=scale*sqrt(ssq);
    return norm;
}
