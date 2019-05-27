//
//  cblas_zher.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zher(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double complex *x, int incx, double complex *A, int ldA)
{
    const double zero=0.0;
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zher","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zher","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zher","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_zher","");
    else if(ldA<n)
        cblas_xerbla(8,"cblas_zher","");
    else if((n>0)&&(alpha!=zero))
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(order==CblasColMajor)
        {
            if(uplo==CblasUpper)
            {
                for(int j=0;j<n;j++)
                {
                    double complex t=alpha*conj(x[j*incx]);
                    for(int i=0;i<j;i++)
                        A[i]+=x[i*incx]*t;
                    A[j]=creal(A[j])+creal(x[j*incx]*t);
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double complex t=alpha*conj(x[j*incx]);
                    A[j]=creal(A[j])+creal(t*x[j*incx]);
                    for(int i=j+1;i<n;i++)
                        A[i]+=x[i*incx]*t;
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
                    double complex t=alpha*x[i*incx];
                    for(int j=0;j<i;j++)
                        A[j]+=conj(x[j*incx])*t;
                    A[i]=creal(A[i])+creal(conj(x[i*incx])*t);
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double complex t=alpha*x[i*incx];
                    A[i]=creal(A[i])+creal(t*conj(x[i*incx]));
                    for(int j=i+1;j<n;j++)
                        A[j]+=conj(x[j*incx])*t;
                    A+=ldA;
                }
            }
        }
    }
}
