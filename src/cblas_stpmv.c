//
//  cblas_stpmv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_stpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n, float *A, float *x, int incx)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_stpmv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_stpmv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(3,"cblas_stpmv","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(4,"cblas_stpmv","");
    else if(n<0)
        cblas_xerbla(5,"cblas_stpmv","");
    else if(incx==0)
        cblas_xerbla(8,"cblas_stpmv","");
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
                        A+=j+1;
                    }
                }
                else if(uplo==CblasLower)
                {
                    A+=n*(n+1)/2;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=n-j;
                        for(int i=n-1;i>j;i--)
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
                    A+=n*(n+1)/2;
                    for(int j=n-1;j>=0;j--)
                    {
                        A-=j+1;
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
                            x[j*incx]*=A[0];
                        for(int i=j+1;i<n;i++)
                            x[j*incx]+=A[i-j]*x[i*incx];
                        A+=n-j;
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
                        for(int j=0;j<i;j++)
                            x[j*incx]+=x[i*incx]*A[j];
                        if(nounit)
                            x[i*incx]*=A[i];
                        A+=i+1;
                    }
                }
                else if(uplo==CblasUpper)
                {
                    A+=n*(n+1)/2;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=n-i;
                        for(int j=n-1;j>i;j--)
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
                    A+=n*(n+1)/2;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=i+1;
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
                            x[i*incx]*=A[0];
                        for(int j=i+1;j<n;j++)
                            x[i*incx]+=A[j-i]*x[j*incx];
                        A+=n-i;
                    }
                }
            }
        }
    }
}
