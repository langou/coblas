//
//  cblas_snrm2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

float cblas_snrm2(int n, float *x, int incx)
{
    const float one=1.0f;
    const float zero=0.0f;
    float norm=0.0f;
    if(n==1)
    {
        norm=fabsf(x[0]);
    }
    else if(n>1)
    {
        float scale=zero;
        float ssq=one;
        for(int i=0;i<n*incx;i+=incx)
        {
            if(x[i]!=zero)
            {
                float a=fabsf(x[i]);
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
        norm=scale*sqrtf(ssq);
    }
    return norm;
}
