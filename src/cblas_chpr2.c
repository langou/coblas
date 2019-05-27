//
//  cblas_chpr2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_chpr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float complex *Alpha, float complex *x, int incx, float complex *y, int incy, float complex *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_chpr2","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_chpr2","");
    else if(n<0)
        cblas_xerbla(3,"cblas_chpr2","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_chpr2","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_chpr2","");
    else
    {
        const float complex alpha=*Alpha;
        const float complex zero=0.0f;
        if(alpha==zero)
            return;
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
                    float complex tx=conjf(alpha*x[j*incx]);
                    float complex ty=alpha*conjf(y[j*incy]);
                    for(int i=0;i<j;i++)
                        A[i]+=x[i*incx]*ty+y[i*incy]*tx;
                    A[j]=crealf(A[j])+crealf(x[j*incx]*ty+y[j*incy]*tx);
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    float complex tx=conjf(alpha*x[j*incx]);
                    float complex ty=alpha*conjf(y[j*incy]);
                    A[0]=crealf(A[0])+crealf(x[j*incx]*ty+y[j*incy]*tx);
                    for(int i=j+1;i<n;i++)
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
                    float complex tx=alpha*x[i*incx];
                    float complex ty=conjf(alpha)*y[i*incy];
                    for(int j=0;j<i;j++)
                        A[j]+=conjf(x[j*incx])*ty+conjf(y[j*incy])*tx;
                    A[i]=crealf(A[i])+crealf(conjf(x[i*incx])*ty+conjf(y[i*incy])*tx);
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    float complex tx=alpha*x[i*incx];
                    float complex ty=conjf(alpha)*y[i*incy];
                    A[0]=crealf(A[0])+crealf(conjf(x[i*incx])*ty+conjf(y[i*incy])*tx);
                    for(int j=i+1;j<n;j++)
                        A[j-i]+=conjf(x[j*incx])*ty+conjf(y[j*incy])*tx;
                    A+=n-i;
                }
            }
        }
    }
}
