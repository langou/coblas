//
//  cblas_dspr2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dspr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dspr2","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dspr2","");
    else if(n<0)
        cblas_xerbla(3,"cblas_dspr2","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_dspr2","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_dspr2","");
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(order==CblasColMajor)
        {
            if(uplo==CblasUpper)
            {
                for(int j=0;j<n;j++)
                {
                    double tx=alpha*x[j*incx];
                    double ty=alpha*y[j*incy];
                    for(int i=0;i<=j;i++)
                        A[i]+=x[i*incx]*ty+y[i*incy]*tx;
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double tx=alpha*x[j*incx];
                    double ty=alpha*y[j*incy];
                    for(int i=j;i<n;i++)
                        A[i-j]+=x[i*incx]*ty+y[i*incy]*tx;
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
                    double tx=alpha*x[i*incx];
                    double ty=alpha*y[i*incy];
                    for(int j=0;j<=i;j++)
                        A[j]+=x[j*incx]*ty+y[j*incy]*tx;
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double tx=alpha*x[i*incx];
                    double ty=alpha*y[i*incy];
                    for(int j=i;j<n;j++)
                        A[j-i]+=x[j*incx]*ty+y[j*incy]*tx;
                    A+=n-i;
                }
            }
        }
    }
}
