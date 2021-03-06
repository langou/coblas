//
//  cblas_stbmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

static int maxsub(int j, int k)
{
    return (j>k)?j-k:0;
}

static int minadd(int j, int k, int n)
{
    return (j+k<n)?j+k:n;
}

void cblas_stbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n, int k, float *A, int ldA, float *x, int incx)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_stbmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_stbmv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(3,"cblas_stbmv","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(4,"cblas_stbmv","");
    else if(n<0)
        cblas_xerbla(5,"cblas_stbmv","");
    else if(k<0)
        cblas_xerbla(6,"cblas_stbmv","");
    else if(ldA<k+1)
        cblas_xerbla(8,"cblas_stbmv","");
    else if(incx==0)
        cblas_xerbla(10,"cblas_stbmv","");
    else
    {
        const bool nounit=(diag==CblasNonUnit);
        if(incx<0)
            x-=(n-1)*incx;
        if(order==CblasColMajor)
        {
            if(trans==CblasNoTrans)
            {
                if(uplo==CblasUpper)
                {
                    for(int j=0;j<n;j++)
                    {
                        int i0=maxsub(j,k);
                        for(int i=i0;i<j;i++)
                            x[i*incx]+=x[j*incx]*A[k+i-j];
                        if(nounit)
                            x[j*incx]*=A[k];
                        A+=ldA;
                    }
                }
                else if(uplo==CblasLower)
                {
                    A+=n*ldA;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=ldA;
                        int im=minadd(j,k,n-1);
                        for(int i=im;i>j;i--)
                            x[i*incx]+=x[j*incx]*A[i-j];
                        if(nounit)
                            x[j*incx]*=A[0];
                    }
                }
            }
            else if((trans==CblasTrans)||(trans==CblasConjTrans))
            {
                if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[j*incx]*=A[k];
                        int i0=maxsub(j,k);
                        for(int i=j-1;i>=i0;i--)
                            x[j*incx]+=A[k+i-j]*x[i*incx];
                    }
                }
                else if(uplo==CblasLower)
                {
                    for(int j=0;j<n;j++)
                    {
                        if(nounit)
                            x[j*incx]*=A[0];
                        int im=minadd(j,k,n-1);
                        for(int i=j+1;i<=im;i++)
                            x[j*incx]+=A[i-j]*x[i*incx];
                        A+=ldA;
                    }
                }
            }
        }
        else if(order==CblasRowMajor)
        {
            if((trans==CblasTrans)||(trans==CblasConjTrans))
            {
                if(uplo==CblasLower)
                {
                    for(int i=0;i<n;i++)
                    {
                        int j0=maxsub(i,k);
                        for(int j=j0;j<i;j++)
                            x[j*incx]+=x[i*incx]*A[k+j-i];
                        if(nounit)
                            x[i*incx]*=A[k];
                        A+=ldA;
                    }
                }
                else if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        int jn=minadd(i,k,n-1);
                        for(int j=jn;j>i;j--)
                            x[j*incx]+=x[i*incx]*A[j-i];
                        if(nounit)
                            x[i*incx]*=A[0];
                    }
                }
            }
            else if(trans==CblasNoTrans)
            {
                if(uplo==CblasLower)
                {
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[i*incx]*=A[k];
                        int j0=maxsub(i,k);
                        for(int j=i-1;j>=j0;j--)
                            x[i*incx]+=A[k+j-i]*x[j*incx];
                    }
                }
                else if(uplo==CblasUpper)
                {
                    for(int i=0;i<n;i++)
                    {
                        if(nounit)
                            x[i*incx]*=A[0];
                        int jn=minadd(i,k,n-1);
                        for(int j=i+1;j<=jn;j++)
                            x[i*incx]+=A[j-i]*x[j*incx];
                        A+=ldA;
                    }
                }
            }
        }
    }
}
