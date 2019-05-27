//
//  cblas_dsyrk.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dsyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double alpha, double *A, int ldA, double beta, double *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dsyrk","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_dsyrk","");
    else if(n<0)
        cblas_xerbla(4,"cblas_dsyrk","");
    else if(k<0)
        cblas_xerbla(5,"cblas_dsyrk","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_dsyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_dsyrk","");
        else
        {
            const double zero=0.0;
            if(alpha==zero)
            {
                if(uplo==CblasUpper)
                {
                    double *c=C;
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
                    double *c=C;
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
                    double *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<=j;i++)
                            c[i]*=beta;
                        double *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double t=alpha*a[j];
                            for(int i=0;i<=j;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=j;i<n;i++)
                            c[i]*=beta;
                        double *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double t=alpha*a[j];
                            for(int i=j;i<n;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
            }
            else if((tran==CblasTrans)||(tran==CblasConjTrans))
            {
                if(uplo==CblasUpper)
                {
                    double *c=C;
                    double *at=A;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A;
                        for(int i=0;i<=j;i++)
                        {
                            double t=zero;
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
                    double *at=A;
                    double *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A+j*ldA;
                        for(int i=j;i<n;i++)
                        {
                            double t=zero;
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
            cblas_xerbla(8,"cblas_dsyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_dsyrk","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_dsyrk(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_dsyrk","");
    }
}
