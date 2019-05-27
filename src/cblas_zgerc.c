//
//  cblas_zgerc.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zgerc(CBLAS_ORDER order, int m, int n, double complex *Alpha, double complex *x, int incx, double complex *y, int incy, double complex *A, int ldA)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zgerc","");
    else if(m<0)
        cblas_xerbla(2,"cblas_zgerc","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zgerc","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_zgerc","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_zgerc","");
    else if(((order==CblasRowMajor)&&(ldA<n))||((order==CblasColMajor)&&(ldA<m)))
        cblas_xerbla(10,"cblas_zgerc","");
    else
    {
        const double complex alpha=*Alpha;
        if(incx<0)
            x-=(m-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(order==CblasColMajor)
        {
            for(int j=0;j<n;j++)
            {
                for(int i=0;i<m;i++)
                    A[i]+=x[i*incx]*alpha*conj(y[j*incy]);
                A+=ldA;
            }
        }
        else if(order==CblasRowMajor)
        {
            for(int i=0;i<m;i++)
            {
                for(int j=0;j<n;j++)
                    A[j]+=x[i*incx]*alpha*conj(y[j*incy]);
                A+=ldA;
            }
        }
    }
}
