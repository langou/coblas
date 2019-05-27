//
//  cblas_sgemv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int m, int n, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_sgemv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(2,"cblas_sgemv","");
    else if(m<0)
        cblas_xerbla(3,"cblas_sgemv","");
    else if(n<0)
        cblas_xerbla(4,"cblas_sgemv","");
    else if(((order==CblasRowMajor)&&(ldA<n))||((order==CblasColMajor)&&(ldA<m)))
        cblas_xerbla(7,"cblas_sgemv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_sgemv","");
    else if(incy==0)
        cblas_xerbla(12,"cblas_sgemv","");
    else if((m>0)&&(n>0))
    {
        const float one=1.0f;
        const float zero=0.0f;
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
                    float temp=alpha*x[j*incx];
                    for(int i=0;i<m;i++)
                        y[i*incy]+=temp*A[i];
                    A+=ldA;
                }
            }
            else if((trans==CblasTrans)||(trans==CblasConjTrans))
            {
                for(int j=0;j<n;j++)
                {
                    float temp=zero;
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
                    float temp=zero;
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
                    float temp=alpha*x[i*incx];
                    for(int j=0;j<n;j++)
                        y[j*incy]+=temp*A[j];
                    A+=ldA;
                }
            }
        }
    }
}
