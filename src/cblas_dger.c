//
//  cblas_dger.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dger(CBLAS_ORDER order, int m, int n, double alpha, double *x, int incx, double *y, int incy, double *A, int ldA)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dger","");
    else if(m<0)
        cblas_xerbla(2,"cblas_dger","");
    else if(n<0)
        cblas_xerbla(3,"cblas_dger","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_dger","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_dger","");
    else if(((order==CblasRowMajor)&&(ldA<n))||((order==CblasColMajor)&&(ldA<m)))
        cblas_xerbla(10,"cblas_dger","");
    else
    {
        if(incx<0)
            x-=(m-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(order==CblasColMajor)
        {
            for(int j=0;j<n;j++)
            {
                for(int i=0;i<m;i++)
                    A[i]+=x[i*incx]*alpha*y[j*incy];
                A+=ldA;
            }
        }
        else if(order==CblasRowMajor)
        {
            for(int i=0;i<m;i++)
            {
                for(int j=0;j<n;j++)
                    A[j]+=x[i*incx]*alpha*y[j*incy];
                A+=ldA;
            }
        }
    }
}
