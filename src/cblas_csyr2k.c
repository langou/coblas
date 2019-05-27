//
//  cblas_csyr2k.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_csyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, float complex *Alpha, float complex *A, int ldA, float complex *B, int ldB, float complex *Beta, float complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_csyr2k","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans))
        cblas_xerbla(3,"cblas_csyr2k","");
    else if(n<0)
        cblas_xerbla(4,"cblas_csyr2k","");
    else if(k<0)
        cblas_xerbla(5,"cblas_csyr2k","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_csyr2k","");
        else if(ldB<(tran==CblasNoTrans?n:k))
            cblas_xerbla(10,"cblas_csyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_csyr2k","");
        else
        {
            const float complex alpha=*Alpha;
            const float complex beta=*Beta;
            const float complex zero=0.0f;
            if(alpha==zero)
            {
                if(uplo==CblasUpper)
                {
                    float complex *c=C;
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
                    float complex *c=C;
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
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<=j;i++)
                            c[i]*=beta;
                        float complex *a=A;
                        float complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            float complex s=alpha*b[j];
                            float complex t=alpha*a[j];
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
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=j;i<n;i++)
                            c[i]*=beta;
                        float complex *a=A;
                        float complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            float complex s=alpha*b[j];
                            float complex t=alpha*a[j];
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
                    float complex *c=C;
                    float complex *at=A;
                    float complex *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        float complex *a=A;
                        float complex *b=B;
                        for(int i=0;i<=j;i++)
                        {
                            float complex s=zero;
                            float complex t=zero;
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
                    float complex *c=C;
                    float complex *at=A;
                    float complex *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        float complex *a=A+j*ldA;
                        float complex *b=B+j*ldB;
                        for(int i=j;i<n;i++)
                        {
                            float complex s=zero;
                            float complex t=zero;
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
            cblas_xerbla(8,"cblas_csyr2k","");
        else if(ldB<(tran==CblasNoTrans?k:n))
            cblas_xerbla(10,"cblas_csyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_csyr2k","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_csyr2k(CblasColMajor,Uplo,Tran,n,k,Alpha,A,ldA,B,ldB,Beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_csyr2k","");
    }
}
