//
//  cblas_zgemv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int m, int n, double complex *Alpha, double complex *A, int ldA, double complex *x, int incx, double complex *Beta, double complex *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_zgemv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(2,"cblas_zgemv","");
    else if(m<0)
        cblas_xerbla(3,"cblas_zgemv","");
    else if(n<0)
        cblas_xerbla(4,"cblas_zgemv","");
    else if(((order==CblasRowMajor)&&(ldA<n))||((order==CblasColMajor)&&(ldA<m)))
        cblas_xerbla(7,"cblas_zgemv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_zgemv","");
    else if(incy==0)
        cblas_xerbla(12,"cblas_zgemv","");
    else if((m>0)&&(n>0))
    {
        const double complex one=1.0;
        const double complex zero=0.0;
        const double complex alpha=*Alpha;
        const double complex beta=*Beta;
        const int lenx=(trans==CblasNoTrans)?n:m;
        const int leny=(trans==CblasNoTrans)?m:n;
        if(incx<0)
            x-=(lenx-1)*incx;
        if(incy<0)
            y-=(leny-1)*incy;
        if(beta==zero)
        {
            for(int i=0;i<leny;i++)
                y[i*incy]=zero;
        }
        else if(beta!=one)
        {
            for(int i=0;i<leny;i++)
                y[i*incy]*=beta;
        }
        if(order==CblasColMajor)
        {
            if(trans==CblasNoTrans)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=alpha*x[j*incx];
                    for(int i=0;i<m;i++)
                        y[i*incy]+=temp*A[i];
                    A+=ldA;
                }
            }
            else if(trans==CblasTrans)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=zero;
                    for(int i=0;i<m;i++)
                        temp+=A[i]*x[i*incx];
                    y[j*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
            else if(trans==CblasConjTrans)
            {
                for(int j=0;j<n;j++)
                {
                    double complex temp=zero;
                    for(int i=0;i<m;i++)
                        temp+=conj(A[i])*x[i*incx];
                    y[j*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
        }
        else if(order==CblasRowMajor)
        {
            if(trans==CblasNoTrans)
            {
                for(int i=0;i<m;i++)
                {
                    double complex temp=zero;
                    for(int j=0;j<n;j++)
                        temp+=A[j]*x[j*incx];
                    y[i*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
            else if(trans==CblasTrans)
            {
                for(int i=0;i<m;i++)
                {
                    double complex temp=alpha*x[i*incx];
                    for(int j=0;j<n;j++)
                        y[j*incy]+=temp*A[j];
                    A+=ldA;
                }
            }
            else if(trans==CblasConjTrans)
            {
                for(int i=0;i<m;i++)
                {
                    double complex temp=alpha*x[i*incx];
                    for(int j=0;j<n;j++)
                        y[j*incy]+=temp*conj(A[j]);
                    A+=ldA;
                }
            }
        }
    }
}
