//
//  cblas_csscal.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_csscal(int n, float alpha, float complex *x, int incx)
{
    if(incx==1)
        for(int i=0;i<n;i++)
            x[i]*=alpha;
    else
        for(int i=0;i<n*incx;i+=incx)
            x[i]*=alpha;
}
