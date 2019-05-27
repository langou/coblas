//
//  cblas_dsyr2k.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dsyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_dsyr2k","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_dsyr2k","");
    else if(n<0)
        cblas_xerbla(4,"cblas_dsyr2k","");
    else if(k<0)
        cblas_xerbla(5,"cblas_dsyr2k","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_dsyr2k","");
        else if(ldB<(tran==CblasNoTrans?n:k))
            cblas_xerbla(10,"cblas_dsyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_dsyr2k","");
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
                        double *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double s=alpha*b[j];
                            double t=alpha*a[j];
                            for(int i=0;i<=j;i++)
                                c[i]+=s*a[i]+t*b[i];
                            a+=ldA;
                            b+=ldB;
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
                        double *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double s=alpha*b[j];
                            double t=alpha*a[j];
                            for(int i=j;i<n;i++)
                                c[i]+=s*a[i]+t*b[i];
                            a+=ldA;
                            b+=ldB;
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
                    double *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A;
                        double *b=B;
                        for(int i=0;i<=j;i++)
                        {
                            double s=zero;
                            double t=zero;
                            for(int l=0;l<k;l++)
                            {
                                s+=b[l]*at[l];
                                t+=a[l]*bt[l];
                            }
                            c[i]=alpha*t+alpha*s+beta*c[i];
                            a+=ldA;
                            b+=ldB;
                        }
                        at+=ldA;
                        bt+=ldB;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double *c=C;
                    double *at=A;
                    double *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A+j*ldA;
                        double *b=B+j*ldB;
                        for(int i=j;i<n;i++)
                        {
                            double s=zero;
                            double t=zero;
                            for(int l=0;l<k;l++)
                            {
                                s+=b[l]*at[l];
                                t+=a[l]*bt[l];
                            }
                            c[i]=alpha*t+alpha*s+beta*c[i];
                            a+=ldA;
                            b+=ldB;
                        }
                        at+=ldA;
                        bt+=ldB;
                        c+=ldC;
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(tran==CblasNoTrans?k:n))
            cblas_xerbla(8,"cblas_dsyr2k","");
        else if(ldB<(tran==CblasNoTrans?k:n))
            cblas_xerbla(10,"cblas_dsyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_dsyr2k","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_dsyr2k(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_dsyr2k","");
    }
}
