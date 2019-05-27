//
//  cblas_drotg.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

void cblas_drotg(double *A, double *B, double *C, double *S)
{
    const double one=1.0;
    const double zero=0.0;
    double a=*A;
    double b=*B;
    double c=*C;
    double s=*S;
    double r=zero;
    double z=zero;
    double roe=(fabs(a)>fabs(b))?a:b;
    double scale=fabs(a)+fabs(b);
    if(scale==zero)
    {
        c=one;
        s=zero;
    }
    else
    {
        double ta=(a/scale);
        double tb=(b/scale);
        if(roe>zero)
            r=scale*sqrt(ta*ta+tb*tb);
        else
            r=-scale*sqrt(ta*ta+tb*tb);
        c=a/r;
        s=b/r;
        if(fabs(a)>fabs(b))
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
