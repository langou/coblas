//
//  cblas_ctrmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_ctrmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n, float complex *A, int ldA, float complex *x, int incx)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_ctrmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_ctrmv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(3,"cblas_ctrmv","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(4,"cblas_ctrmv","");
    else if(n<0)
        cblas_xerbla(5,"cblas_ctrmv","");
    else if(ldA<n)
        cblas_xerbla(7,"cblas_ctrmv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_ctrmv","");
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
                        for(int i=0;i<j;i++)
                            x[i*incx]+=x[j*incx]*A[i];
                        if(nounit)
                            x[j*incx]*=A[j];
                        A+=ldA;
                    }
                }
                else if(uplo==CblasLower)
                {
                    A+=n*ldA;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=ldA;
                        for(int i=n-1;i>j;i--)
                            x[i*incx]+=x[j*incx]*A[i];
                        if(nounit)
                            x[j*incx]*=A[j];
                    }
                }
            }
            else if(trans==CblasTrans)
            {
                if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[j*incx]*=A[j];
                        for(int i=j-1;i>=0;i--)
                            x[j*incx]+=A[i]*x[i*incx];
                    }
                }
                else if(uplo==CblasLower)
                {
                    for(int j=0;j<n;j++)
                    {
                        if(nounit)
                            x[j*incx]*=A[j];
                        for(int i=j+1;i<n;i++)
                            x[j*incx]+=A[i]*x[i*incx];
                        A+=ldA;
                    }
                }
            }
            else if(trans==CblasConjTrans)
            {
                if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[j*incx]*=conjf(A[j]);
                        for(int i=j-1;i>=0;i--)
                            x[j*incx]+=conjf(A[i])*x[i*incx];
                    }
                }
                else if(uplo==CblasLower)
                {
                    for(int j=0;j<n;j++)
                    {
                        if(nounit)
                            x[j*incx]*=conjf(A[j]);
                        for(int i=j+1;i<n;i++)
                            x[j*incx]+=conjf(A[i])*x[i*incx];
                        A+=ldA;
                    }
                }
            }
        }
        else if(order==CblasRowMajor)
        {
            if(trans==CblasTrans)
            {
                if(uplo==CblasLower)
                {
                    for(int i=0;i<n;i++)
                    {
                        for(int j=0;j<i;j++)
                            x[j*incx]+=x[i*incx]*A[j];
                        if(nounit)
                            x[i*incx]*=A[i];
                        A+=ldA;
                    }
                }
                else if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        for(int j=n-1;j>i;j--)
                            x[j*incx]+=x[i*incx]*A[j];
                        if(nounit)
                            x[i*incx]*=A[i];
                    }
                }
            }
            if(trans==CblasConjTrans)
            {
                if(uplo==CblasLower)
                {
                    for(int i=0;i<n;i++)
                    {
                        for(int j=0;j<i;j++)
                            x[j*incx]+=x[i*incx]*conjf(A[j]);
                        if(nounit)
                            x[i*incx]*=conjf(A[i]);
                        A+=ldA;
                    }
                }
                else if(uplo==CblasUpper)
                {
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        for(int j=n-1;j>i;j--)
                            x[j*incx]+=x[i*incx]*conjf(A[j]);
                        if(nounit)
                            x[i*incx]*=conjf(A[i]);
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
                            x[i*incx]*=A[i];
                        for(int j=i-1;j>=0;j--)
                            x[i*incx]+=A[j]*x[j*incx];
                    }
                }
                else if(uplo==CblasUpper)
                {
                    for(int i=0;i<n;i++)
                    {
                        if(nounit)
                            x[i*incx]*=A[i];
                        for(int j=i+1;j<n;j++)
                            x[i*incx]+=A[j]*x[j*incx];
                        A+=ldA;
                    }
                }
            }
        }
    }
}
