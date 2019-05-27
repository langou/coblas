//
//  cblas_zsyrk.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zsyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double complex *Alpha, double complex *A, int ldA, double complex *Beta, double complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zsyrk","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans))
        cblas_xerbla(3,"cblas_zsyrk","");
    else if(n<0)
        cblas_xerbla(4,"cblas_zsyrk","");
    else if(k<0)
        cblas_xerbla(5,"cblas_zsyrk","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_zsyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_zsyrk","");
        else
        {
            const double complex alpha=*Alpha;
            const double complex beta=*Beta;
            const double complex zero=0.0;
            if(alpha==zero)
            {
                if(uplo==CblasUpper)
                {
                    double complex *c=C;
                    if(beta==zero)
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=0;i<=j;i++)
                                c[i]=zero;
                            c+=ldC;
                        }
                    }
                    else
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=0;i<=j;i++)
                                c[i]*=beta;
                            c+=ldC;
                        }
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *c=C;
                    if(beta==zero)
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=j;i<n;i++)
                                c[i]=zero;
                            c+=ldC;
                        }
                    }
                    else
                    {
                        for(int j=0;j<n;j++)
                        {
                            for(int i=j;i<n;i++)
                                c[i]*=beta;
                            c+=ldC;
                        }
                    }
                }
            }
            else if(tran==CblasNoTrans)
            {
                if(uplo==CblasUpper)
                {
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<=j;i++)
                            c[i]*=beta;
                        double complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double complex t=alpha*a[j];
                            for(int i=0;i<=j;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=j;i<n;i++)
                            c[i]*=beta;
                        double complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double complex t=alpha*a[j];
                            for(int i=j;i<n;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
            }
            else if(tran==CblasTrans)
            {
                if(uplo==CblasUpper)
                {
                    double complex *c=C;
                    double complex *at=A;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A;
                        for(int i=0;i<=j;i++)
                        {
                            double complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=a[l]*at[l];
                            c[i]=alpha*t+beta*c[i];
                            a+=ldA;
                        }
                        at+=ldA;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *at=A;
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A+j*ldA;
                        for(int i=j;i<n;i++)
                        {
                            double complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=a[l]*at[l];
                            c[i]=alpha*t+beta*c[i];
                            a+=ldA;
                        }
                        at+=ldA;
                        c+=ldC;
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(tran==CblasNoTrans?k:n))
            cblas_xerbla(8,"cblas_zsyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_zsyrk","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_zsyrk(CblasColMajor,Uplo,Tran,n,k,Alpha,A,ldA,Beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_zsyrk","");
    }
}
