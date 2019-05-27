//
//  cblas_zhpr.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zhpr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double complex *x, int incx, double complex *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zhpr","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zhpr","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zhpr","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_zhpr","");
    else
    {
        const double zero=0.0;
        if(alpha==zero)
            return;
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
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double complex t=alpha*conj(x[j*incx]);
                    A[0]=creal(A[0])+creal(t*x[j*incx]);
                    for(int i=j+1;i<n;i++)
                        A[i-j]+=x[i*incx]*t;
                    A+=n-j;
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
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double complex t=alpha*x[i*incx];
                    A[0]=creal(A[0])+creal(t*conj(x[i*incx]));
                    for(int j=i+1;j<n;j++)
                        A[j-i]+=conj(x[j*incx])*t;
                    A+=n-i;
                }
            }
        }
    }
}
