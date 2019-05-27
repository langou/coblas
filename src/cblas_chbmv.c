//
//  cblas_chbmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_chbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, float complex *Alpha, float complex *A, int ldA, float complex *x, int incx, float complex *Beta, float complex *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_chbmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_chbmv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_chbmv","");
    else if(k<0)
        cblas_xerbla(4,"cblas_chbmv","");
    else if(ldA<k+1)
        cblas_xerbla(7,"cblas_chbmv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_chbmv","");
    else if(incy==0)
        cblas_xerbla(12,"cblas_chbmv","");
    else
    {
        const float complex one=1.0f;
        const float complex zero=0.0f;
        const float complex alpha=*Alpha;
        const float complex beta=*Beta;
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
                    float complex temp=alpha*x[j*incx];
                    float complex sum=zero;
                    int i0=(j>k)?j-k:0;
                    for(int i=i0;i<j;i++)
                    {
                        y[i*incy]+=temp*A[k-j+i];
                        sum+=conjf(A[k-j+i])*x[i*incx];
                    }
                    y[j*incy]+=temp*crealf(A[k])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    float complex temp=alpha*x[j*incx];
                    float complex sum=zero;
                    int im=(j+k<n)?j+k+1:n;
                    y[j*incy]+=temp*crealf(A[0]);
                    for(int i=j+1;i<im;i++)
                    {
                        y[i*incy]+=temp*A[i-j];
                        sum+=conjf(A[i-j])*x[i*incx];
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
                    float complex temp=alpha*x[i*incx];
                    float complex sum=zero;
                    int j0=(i>k)?i-k:0;
                    for(int j=j0;j<i;j++)
                    {
                        y[j*incy]+=temp*conjf(A[k-i+j]);
                        sum+=A[k-i+j]*x[j*incx];
                    }
                    y[i*incy]+=temp*crealf(A[k])+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    float complex temp=alpha*x[i*incx];
                    float complex sum=zero;
                    int jn=(i+k<n)?i+k+1:n;
                    y[i*incy]+=temp*crealf(A[0]);
                    for(int j=i+1;j<jn;j++)
                    {
                        y[j*incy]+=temp*conjf(A[j-i]);
                        sum+=A[j-i]*x[j*incx];
                    }
                    y[i*incy]+=alpha*sum;
                    A+=ldA;
                }
            }
        }
    }
}
