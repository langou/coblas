//
//  cblas_dsyr2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dsyr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *A, int ldA)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dsyr2","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dsyr2","");
    else if(n<0)
        cblas_xerbla(3,"cblas_dsyr2","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_dsyr2","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_dsyr2","");
    else if(ldA<n)
        cblas_xerbla(10,"cblas_dsyr2","");
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(order==CblasColMajor)
        {
            if(uplo==CblasUpper)
            {
                for(int j=0;j<n;j++)
                {
                    double tx=alpha*x[j*incx];
                    double ty=alpha*y[j*incy];
                    for(int i=0;i<=j;i++)
                        A[i]+=x[i*incx]*ty+y[i*incy]*tx;
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double tx=alpha*x[j*incx];
                    double ty=alpha*y[j*incy];
                    for(int i=j;i<n;i++)
                        A[i]+=x[i*incx]*ty+y[i*incy]*tx;
                    A+=ldA;
                }
            }
        }
        else if(order==CblasRowMajor)
        {
            if(uplo==CblasLower)
            {
                for(int i=0;i<n;i++)
                {
                    double tx=alpha*x[i*incx];
                    double ty=alpha*y[i*incy];
                    for(int j=0;j<=i;j++)
                        A[j]+=x[j*incx]*ty+y[j*incy]*tx;
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double tx=alpha*x[i*incx];
                    double ty=alpha*y[i*incy];
                    for(int j=i;j<n;j++)
                        A[j]+=x[j*incx]*ty+y[j*incy]*tx;
                    A+=ldA;
                }
            }
        }
    }
}
