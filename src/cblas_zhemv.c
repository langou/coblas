//
//  cblas_zhemv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zhemv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double complex *Alpha, double complex *A, int ldA, double complex *x, int incx, double complex *Beta, double complex *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zhemv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zhemv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zhemv","");
    else if(ldA<n)
        cblas_xerbla(6,"cblas_zhemv","");
    else if(incx==0)
        cblas_xerbla(8,"cblas_zhemv","");
    else if(incy==0)
        cblas_xerbla(11,"cblas_zhemv","");
    else if(n>0)
    {
        const double complex one=1.0;
        const double complex zero=0.0;
        const double complex alpha=*Alpha;
        const double complex beta=*Beta;
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(beta==zero)
        {
            for(int i=0;i<n;i++)
                y[i*incy]=zero;
        }
        else if(beta!=one)
        {
            for(int i=0;i<n;i++)
                y[i*incy]*=beta;
        }
        if(order==CblasColMajor)
        {
            if(uplo==CblasUpper)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=alpha*x[j*incx];
                    double complex sum=zero;
                    for(int i=0;i<j;i++)
                    {
                        y[i*incy]+=temp*A[i];
                        sum+=conj(A[i])*x[i*incx];
                    }
                    y[j*incy]+=temp*creal(A[j])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=alpha*x[j*incx];
                    double complex sum=zero;
                    y[j*incy]+=temp*creal(A[j]);
                    for(int i=j+1;i<n;i++)
                    {
                        y[i*incy]+=temp*A[i];
                        sum+=conj(A[i])*x[i*incx];
                    }
                    y[j*incy]+=alpha*sum;
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
                    double complex temp=alpha*x[i*incx];
                    double complex sum=zero;
                    for(int j=0;j<i;j++)
                    {
                        y[j*incy]+=temp*conj(A[j]);
                        sum+=A[j]*x[j*incx];
                    }
                    y[i*incy]+=temp*creal(A[i])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double complex temp=alpha*x[i*incx];
                    double complex sum=zero;
                    y[i*incy]+=temp*creal(A[i]);
                    for(int j=i+1;j<n;j++)
                    {
                        y[j*incy]+=temp*conj(A[j]);
                        sum+=A[j]*x[j*incx];
                    }
                    y[i*incy]+=alpha*sum;
                    A+=ldA;
                }
            }
        }
    }
}
