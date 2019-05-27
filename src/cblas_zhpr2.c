//
//  cblas_zhpr2.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zhpr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double complex *Alpha, double complex *x, int incx, double complex *y, int incy, double complex *A)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zhpr2","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zhpr2","");
    else if(n<0)
        cblas_xerbla(3,"cblas_zhpr2","");
    else if(incx==0)
        cblas_xerbla(6,"cblas_zhpr2","");
    else if(incy==0)
        cblas_xerbla(8,"cblas_zhpr2","");
    else
    {
        const double complex alpha=*Alpha;
        const double complex zero=0.0f;
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
                    double complex tx=conj(alpha*x[j*incx]);
                    double complex ty=alpha*conj(y[j*incy]);
                    for(int i=0;i<j;i++)
                        A[i]+=x[i*incx]*ty+y[i*incy]*tx;
                    A[j]=creal(A[j])+creal(x[j*incx]*ty+y[j*incy]*tx);
                    A+=j+1;
                }
            }
            else if(uplo==CblasLower)
            {
                for(int j=0;j<n;j++)
                {
                    double complex tx=conj(alpha*x[j*incx]);
                    double complex ty=alpha*conj(y[j*incy]);
                    A[0]=creal(A[0])+creal(x[j*incx]*ty+y[j*incy]*tx);
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
                    double complex tx=alpha*x[i*incx];
                    double complex ty=conj(alpha)*y[i*incy];
                    for(int j=0;j<i;j++)
                        A[j]+=conj(x[j*incx])*ty+conj(y[j*incy])*tx;
                    A[i]=creal(A[i])+creal(conj(x[i*incx])*ty+conj(y[i*incy])*tx);
                    A+=i+1;
                }
            }
            else if(uplo==CblasUpper)
            {
                for(int i=0;i<n;i++)
                {
                    double complex tx=alpha*x[i*incx];
                    double complex ty=conj(alpha)*y[i*incy];
                    A[0]=creal(A[0])+creal(conj(x[i*incx])*ty+conj(y[i*incy])*tx);
                    for(int j=i+1;j<n;j++)
                        A[j-i]+=conj(x[j*incx])*ty+conj(y[j*incy])*tx;
                    A+=n-i;
                }
            }
        }
    }
}
