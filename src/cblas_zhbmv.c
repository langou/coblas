//
//  cblas_zhbmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zhbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, double complex *Alpha, double complex *A, int ldA, double complex *x, int incx, double complex *Beta, double complex *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zhbmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zhbmv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zhbmv","");
    else if(k<0)
        cblas_xerbla(4,"cblas_zhbmv","");
    else if(ldA<k+1)
        cblas_xerbla(7,"cblas_zhbmv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_zhbmv","");
    else if(incy==0)
        cblas_xerbla(12,"cblas_zhbmv","");
    else
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
                    int i0=(j>k)?j-k:0;
                    for(int i=i0;i<j;i++)
                    {
                        y[i*incy]+=temp*A[k-j+i];
                        sum+=conj(A[k-j+i])*x[i*incx];
                    }
                    y[j*incy]+=temp*creal(A[k])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=alpha*x[j*incx];
                    double complex sum=zero;
                    int im=(j+k<n)?j+k+1:n;
                    y[j*incy]+=temp*creal(A[0]);
                    for(int i=j+1;i<im;i++)
                    {
                        y[i*incy]+=temp*A[i-j];
                        sum+=conj(A[i-j])*x[i*incx];
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
                    int j0=(i>k)?i-k:0;
                    for(int j=j0;j<i;j++)
                    {
                        y[j*incy]+=temp*conj(A[k-i+j]);
                        sum+=A[k-i+j]*x[j*incx];
                    }
                    y[i*incy]+=temp*creal(A[k])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double complex temp=alpha*x[i*incx];
                    double complex sum=zero;
                    int jn=(i+k<n)?i+k+1:n;
                    y[i*incy]+=temp*creal(A[0]);
                    for(int j=i+1;j<jn;j++)
                    {
                        y[j*incy]+=temp*conj(A[j-i]);
                        sum+=A[j-i]*x[j*incx];
                    }
                    y[i*incy]+=alpha*sum;
                    A+=ldA;
                }
            }
        }
    }
}
