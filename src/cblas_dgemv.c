//
//  cblas_dgemv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int m, int n, double alpha, double *A, int ldA, double *x, int incx, double beta, double *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_dgemv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(2,"cblas_dgemv","");
    else if(m<0)
        cblas_xerbla(3,"cblas_dgemv","");
    else if(n<0)
        cblas_xerbla(4,"cblas_dgemv","");
    else if(((order==CblasRowMajor)&&(ldA<n))||((order==CblasColMajor)&&(ldA<m)))
        cblas_xerbla(7,"cblas_dgemv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_dgemv","");
    else if(incy==0)
        cblas_xerbla(12,"cblas_dgemv","");
    else if((m>0)&&(n>0))
    {
        const double one=1.0;
        const double zero=0.0;
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
                    double temp=alpha*x[j*incx];
                    for(int i=0;i<m;i++)
                        y[i*incy]+=temp*A[i];
                    A+=ldA;
                }
            }
            else if((trans==CblasTrans)||(trans==CblasConjTrans))
            {
                for(int j=0;j<n;j++)
                {
                    double temp=zero;
                    for(int i=0;i<m;i++)
                        temp+=A[i]*x[i*incx];
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
                    double temp=zero;
                    for(int j=0;j<n;j++)
                        temp+=A[j]*x[j*incx];
                    y[i*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
            else if((trans==CblasTrans)||(trans==CblasConjTrans))
            {
                for(int i=0;i<m;i++)
                {
                    double temp=alpha*x[i*incx];
                    for(int j=0;j<n;j++)
                        y[j*incy]+=temp*A[j];
                    A+=ldA;
                }
            }
        }
    }
}
