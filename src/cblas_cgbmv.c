//
//  cblas_cgbmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_cgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int m, int n, int kl, int ku, float complex *Alpha, float complex *A, int ldA, float complex *x, int incx, float complex *Beta, float complex *y, int incy)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_cgbmv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(2,"cblas_cgbmv","");
    else if(m<0)
        cblas_xerbla(3,"cblas_cgbmv","");
    else if(n<0)
        cblas_xerbla(4,"cblas_cgbmv","");
    else if(kl<0)
        cblas_xerbla(5,"cblas_cgbmv","");
    else if(ku<0)
        cblas_xerbla(6,"cblas_cgbmv","");
    else if(ldA<ku+kl+1)
        cblas_xerbla(9,"cblas_cgbmv","");
    else if(incx==0)
        cblas_xerbla(11,"cblas_cgbmv","");
    else if(incy==0)
        cblas_xerbla(14,"cblas_cgbmv","");
    else if((m>0)&&(n>0))
    {
        const float complex one=1.0f;
        const float complex zero=0.0f;
        const float complex alpha=*Alpha;
        const float complex beta=*Beta;
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
                    float complex temp=alpha*x[j*incx];
                    int i0=(j>ku)?j-ku:0;
                    int im=(j+kl<m)?j+kl+1:m;
                    for(int i=i0;i<im;i++)
                        y[i*incy]+=temp*A[ku-j+i];
                    A+=ldA;
                }
            }
            else if(trans==CblasTrans)
            {
                for(int j=0;j<n;j++)
                {
                    float complex temp=zero;
                    int i0=(j>ku)?j-ku:0;
                    int im=(j+kl<m)?j+kl+1:m;
                    for(int i=i0;i<im;i++)
                        temp+=A[ku-j+i]*x[i*incx];
                    y[j*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
            else if(trans==CblasConjTrans)
            {
                for(int j=0;j<n;j++)
                {
                    float complex temp=zero;
                    int i0=(j>ku)?j-ku:0;
                    int im=(j+kl<m)?j+kl+1:m;
                    for(int i=i0;i<im;i++)
                        temp+=conjf(A[ku-j+i])*x[i*incx];
                    y[j*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
        }
        else if(order==CblasRowMajor)
        {
            if(trans==CblasTrans)
            {
                for(int i=0;i<m;i++)
                {
                    float complex temp=alpha*x[i*incx];
                    int j0=(i>kl)?i-kl:0;
                    int jn=(i+ku<n)?i+ku+1:n;
                    for(int j=j0;j<jn;j++)
                        y[j*incy]+=temp*A[kl-i+j];
                    A+=ldA;
                }
            }
            if(trans==CblasConjTrans)
            {
                for(int i=0;i<m;i++)
                {
                    float complex temp=alpha*x[i*incx];
                    int j0=(i>kl)?i-kl:0;
                    int jn=(i+ku<n)?i+ku+1:n;
                    for(int j=j0;j<jn;j++)
                        y[j*incy]+=temp*conjf(A[kl-i+j]);
                    A+=ldA;
                }
            }
            else if(trans==CblasNoTrans)
            {
                for(int i=0;i<m;i++)
                {
                    float complex temp=zero;
                    int j0=(i>kl)?i-kl:0;
                    int jn=(i+ku<n)?i+ku+1:n;
                    for(int j=j0;j<jn;j++)
                        temp+=A[kl-i+j]*x[j*incx];
                    y[i*incy]+=alpha*temp;
                    A+=ldA;
                }
            }
        }
    }
}
