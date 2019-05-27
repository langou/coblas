//
//  cblas_sspmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_sspmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *A, float *x, int incx, float beta, float *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_sspmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_sspmv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_sspmv","");
    else if(incx==0)
        cblas_xerbla(7,"cblas_sspmv","");
    else if(incy==0)
        cblas_xerbla(10,"cblas_sspmv","");
    else
    {
        const float one=1.0f;
        const float zero=0.0f;
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
                    float temp=alpha*x[j*incx];
                    float sum=zero;
                    for(int i=0;i<j;i++)
                    {
                        y[i*incy]+=temp*A[i];
                        sum+=A[i]*x[i*incx];
                    }
                    y[j*incy]+=temp*A[j]+alpha*sum;
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    float temp=alpha*x[j*incx];
                    float sum=zero;
                    y[j*incy]+=temp*A[0];
                    for(int i=j+1;i<n;i++)
                    {
                        y[i*incy]+=temp*A[i-j];
                        sum+=A[i-j]*x[i*incx];
                    }
                    y[j*incy]+=alpha*sum;
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
                    float temp=alpha*x[i*incx];
                    float sum=zero;
                    for(int j=0;j<i;j++)
                    {
                        y[j*incy]+=temp*A[j];
                        sum+=A[j]*x[j*incx];
                    }
                    y[i*incy]+=temp*A[i]+alpha*sum;
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    float temp=alpha*x[i*incx];
                    float sum=zero;
                    y[i*incy]+=temp*A[0];
                    for(int j=i+1;j<n;j++)
                    {
                        y[j*incy]+=temp*A[j-i];
                        sum+=A[j-i]*x[j*incx];
                    }
                    y[i*incy]+=alpha*sum;
                    A+=n-i;
                }
            }
        }
    }
}
