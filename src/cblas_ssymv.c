//
//  cblas_ssymv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_ssymv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_ssymv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_ssymv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_ssymv","");
    else if(ldA<n)
        cblas_xerbla(6,"cblas_ssymv","");
    else if(incx==0)
        cblas_xerbla(8,"cblas_ssymv","");
    else if(incy==0)
        cblas_xerbla(11,"cblas_ssymv","");
    else if(n>0)
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
                    A+=ldA;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    float temp=alpha*x[j*incx];
                    float sum=zero;
                    y[j*incy]+=temp*A[j];
                    for(int i=j+1;i<n;i++)
                    {
                        y[i*incy]+=temp*A[i];
                        sum+=A[i]*x[i*incx];
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
                    float temp=alpha*x[i*incx];
                    float sum=zero;
                    for(int j=0;j<i;j++)
                    {
                        y[j*incy]+=temp*A[j];
                        sum+=A[j]*x[j*incx];
                    }
                    y[i*incy]+=temp*A[i]+alpha*sum;
                    A+=ldA;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    float temp=alpha*x[i*incx];
                    float sum=zero;
                    y[i*incy]+=temp*A[i];
                    for(int j=i+1;j<n;j++)
                    {
                        y[j*incy]+=temp*A[j];
                        sum+=A[j]*x[j*incx];
                    }
                    y[i*incy]+=alpha*sum;
                    A+=ldA;
                }
            }
        }
    }
}
