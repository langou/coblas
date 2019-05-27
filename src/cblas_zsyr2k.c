//
//  cblas_zsyr2k.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zsyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double complex *Alpha, double complex *A, int ldA, double complex *B, int ldB, double complex *Beta, double complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zsyr2k","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans))
        cblas_xerbla(3,"cblas_zsyr2k","");
    else if(n<0)
        cblas_xerbla(4,"cblas_zsyr2k","");
    else if(k<0)
        cblas_xerbla(5,"cblas_zsyr2k","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_zsyr2k","");
        else if(ldB<(tran==CblasNoTrans?n:k))
            cblas_xerbla(10,"cblas_zsyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_zsyr2k","");
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
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double complex s=alpha*b[j];
                            double complex t=alpha*a[j];
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
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=j;i<n;i++)
                            c[i]*=beta;
                        double complex *a=A;
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double complex s=alpha*b[j];
                            double complex t=alpha*a[j];
                            for(int i=j;i<n;i++)
                                c[i]+=s*a[i]+t*b[i];
                            a+=ldA;
                            b+=ldB;
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
                    double complex *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A;
                        double complex *b=B;
                        for(int i=0;i<=j;i++)
                        {
                            double complex s=zero;
                            double complex t=zero;
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
                    double complex *c=C;
                    double complex *at=A;
                    double complex *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A+j*ldA;
                        double complex *b=B+j*ldB;
                        for(int i=j;i<n;i++)
                        {
                            double complex s=zero;
                            double complex t=zero;
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
            cblas_xerbla(8,"cblas_zsyr2k","");
        else if(ldB<(tran==CblasNoTrans?k:n))
            cblas_xerbla(10,"cblas_zsyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_zsyr2k","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_zsyr2k(CblasColMajor,Uplo,Tran,n,k,Alpha,A,ldA,B,ldB,Beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_zsyr2k","");
    }
}
