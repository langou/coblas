//
//  cblas_srotg.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

void cblas_srotg(float *A, float *B, float *C, float *S)
{
    const float one=1.0f;
    const float zero=0.0f;
    float a=*A;
    float b=*B;
    float c=*C;
    float s=*S;
    float r=zero;
    float z=zero;
    float roe=(fabsf(a)>fabsf(b))?a:b;
    float scale=fabsf(a)+fabsf(b);
    if(scale==zero)
    {
        c=one;
        s=zero;
    }
    else
    {
        float ta=(a/scale);
        float tb=(b/scale);
        if(roe>zero)
            r=scale*sqrtf(ta*ta+tb*tb);
        else
            r=-scale*sqrtf(ta*ta+tb*tb);
        c=a/r;
        s=b/r;
        if(fabsf(a)>fabsf(b))
            z=s;
        else if(c!=zero)
            z=one/c;
        else
            z=one;
    }
    *A=r;
    *B=z;
    *C=c;
    *S=s;
}
