//
//  cblas_dspmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dspmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *A, double *x, int incx, double beta, double *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dspmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dspmv","");
    else if(n<0)
        cblas_xerbla(3,"cblas_dspmv","");
    else if(incx==0)
        cblas_xerbla(7,"cblas_dspmv","");
    else if(incy==0)
        cblas_xerbla(10,"cblas_dspmv","");
    else
    {
        const double one=1.0;
        const double zero=0.0;
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
                    double temp=alpha*x[j*incx];
                    double sum=zero;
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
                    double temp=alpha*x[j*incx];
                    double sum=zero;
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
                    double temp=alpha*x[i*incx];
                    double sum=zero;
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
                    double temp=alpha*x[i*incx];
                    double sum=zero;
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
