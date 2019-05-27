//
//  cblas_dspr.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dspr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dspr","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dspr","");
    else if(n<0)
        cblas_xerbla(3,"cblas_dspr","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_dspr","");
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(order==CblasColMajor)
        {
            if(uplo==CblasUpper)
            {
                for(int j=0;j<n;j++)
                {
                    double t=alpha*x[j*incx];
                    for(int i=0;i<=j;i++)
                        A[i]+=x[i*incx]*t;
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double t=alpha*x[j*incx];
                    for(int i=j;i<n;i++)
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
                    double t=alpha*x[i*incx];
                    for(int j=0;j<=i;j++)
                        A[j]+=x[j*incx]*t;
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double t=alpha*x[i*incx];
                    for(int j=i;j<n;j++)
                        A[j-i]+=x[j*incx]*t;
                    A+=n-i;
                }
            }
        }
    }
}
