//
//  cblas_chpr.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_chpr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float complex *x, int incx, float complex *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_chpr","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_chpr","");
    else if(n<0)
        cblas_xerbla(3,"cblas_chpr","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_chpr","");
    else
    {
        const float zero=0.0f;
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
                    float complex t=alpha*conjf(x[j*incx]);
                    for(int i=0;i<j;i++)
                        A[i]+=x[i*incx]*t;
                    A[j]=crealf(A[j])+crealf(x[j*incx]*t);
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    float complex t=alpha*conjf(x[j*incx]);
                    A[0]=crealf(A[0])+crealf(t*x[j*incx]);
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
                    float complex t=alpha*x[i*incx];
                    for(int j=0;j<i;j++)
                        A[j]+=conjf(x[j*incx])*t;
                    A[i]=crealf(A[i])+crealf(conjf(x[i*incx])*t);
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    float complex t=alpha*x[i*incx];
                    A[0]=crealf(A[0])+crealf(t*conjf(x[i*incx]));
                    for(int j=i+1;j<n;j++)
                        A[j-i]+=conjf(x[j*incx])*t;
                    A+=n-i;
                }
            }
        }
    }
}
