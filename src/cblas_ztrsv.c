//
//  cblas_ztrsv.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_ztrsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n, double complex *A, int ldA, double complex *x, int incx)
{
    if((order!=CblasRowMajor)&&(order!=CblasColMajor))
        cblas_xerbla(1,"cblas_ztrsv","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_ztrsv","");
    else if((trans!=CblasNoTrans)&&(trans!=CblasTrans)&&(trans!=CblasConjTrans))
        cblas_xerbla(3,"cblas_ztrsv","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(4,"cblas_ztrsv","");
    else if(n<0)
        cblas_xerbla(5,"cblas_ztrsv","");
    else if(ldA<n)
        cblas_xerbla(7,"cblas_ztrsv","");
    else if(incx==0)
        cblas_xerbla(9,"cblas_ztrsv","");
    else
    {
        const bool nounit=(diag==CblasNonUnit);
        if(incx<0)
            x-=(n-1)*incx;
        if(order==CblasColMajor)
        {
            if(order==CblasColMajor)
            {
                if(trans==CblasNoTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        A+=n*ldA;
                        for(int j=n-1;j>=0;j--)
                        {
                            A-=ldA;
                            if(nounit)
                                x[j*incx]=x[j*incx]/A[j];
                            for(int i=j-1;i>=0;i--)
                                x[i*incx]-=x[j*incx]*A[i];
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        for(int j=0;j<n;j++)
                        {
                            if(nounit)
                                x[j*incx]=x[j*incx]/A[j];
                            for(int i=j+1;i<n;i++)
                                x[i*incx]-=x[j*incx]*A[i];
                            A+=ldA;
                        }
                    }
                }
                else if(trans==CblasTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=0;i<j;i++)
                                x[j*incx]-=x[i*incx]*A[i];
                            if(nounit)
                                x[j*incx]=x[j*incx]/A[j];
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
                                x[j*incx]-=x[i*incx]*A[i];
                            if(nounit)
                                x[j*incx]=x[j*incx]/A[j];
                        }
                    }
                }
                else if(trans==CblasConjTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=0;i<j;i++)
                                x[j*incx]-=x[i*incx]*conj(A[i]);
                            if(nounit)
                                x[j*incx]=x[j*incx]/conj(A[j]);
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
                                x[j*incx]-=x[i*incx]*conj(A[i]);
                            if(nounit)
                                x[j*incx]=x[j*incx]/conj(A[j]);
                        }
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
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[i*incx]=x[i*incx]/A[i];
                        for(int j=i-1;j>=0;j--)
                            x[j*incx]-=x[i*incx]*A[j];
                    }
                }
                else if(uplo==CblasUpper)
                {
                    for(int i=0;i<n;i++)
                    {
                        if(nounit)
                            x[i*incx]=x[i*incx]/A[i];
                        for(int j=i+1;j<n;j++)
                            x[j*incx]-=x[i*incx]*A[j];
                        A+=ldA;
                    }
                }
            }
            if(trans==CblasConjTrans)
            {
                if(uplo==CblasLower)
                {
                    A+=n*ldA;
                    for(int i=n-1;i>=0;i--)
                    {
                        A-=ldA;
                        if(nounit)
                            x[i*incx]=x[i*incx]/conj(A[i]);
                        for(int j=i-1;j>=0;j--)
                            x[j*incx]-=x[i*incx]*conj(A[j]);
                    }
                }
                else if(uplo==CblasUpper)
                {
                    for(int i=0;i<n;i++)
                    {
                        if(nounit)
                            x[i*incx]=x[i*incx]/conj(A[i]);
                        for(int j=i+1;j<n;j++)
                            x[j*incx]-=x[i*incx]*conj(A[j]);
                        A+=ldA;
                    }
                }
            }
            else if(trans==CblasNoTrans)
            {
                if(uplo==CblasLower)
                {
                    for(int i=0;i<n;i++)
                    {
                        for(int j=0;j<i;j++)
                            x[i*incx]-=x[j*incx]*A[j];
                        if(nounit)
                            x[i*incx]=x[i*incx]/A[i];
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
                            x[i*incx]-=x[j*incx]*A[j];
                        if(nounit)
                            x[i*incx]=x[i*incx]/A[i];
                    }
                }
            }
        }
    }
}
