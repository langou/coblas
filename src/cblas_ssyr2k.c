//
//  cblas_ssyr2k.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_ssyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_ssyr2k","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_ssyr2k","");
    else if(n<0)
        cblas_xerbla(4,"cblas_ssyr2k","");
    else if(k<0)
        cblas_xerbla(5,"cblas_ssyr2k","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_ssyr2k","");
        else if(ldB<(tran==CblasNoTrans?n:k))
            cblas_xerbla(10,"cblas_ssyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_ssyr2k","");
        else
        {
            const float zero=0.0f;
            if(alpha==zero)
            {
                if(uplo==CblasUpper)
                {
                    float *c=C;
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
                    float *c=C;
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
                    float *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<=j;i++)
                            c[i]*=beta;
                        float *a=A;
                        float *b=B;
                        for(int l=0;l<k;l++)
                        {
                            float s=alpha*b[j];
                            float t=alpha*a[j];
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
                    float *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=j;i<n;i++)
                            c[i]*=beta;
                        float *a=A;
                        float *b=B;
                        for(int l=0;l<k;l++)
                        {
                            float s=alpha*b[j];
                            float t=alpha*a[j];
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
                    float *c=C;
                    float *at=A;
                    float *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A;
                        float *b=B;
                        for(int i=0;i<=j;i++)
                        {
                            float s=zero;
                            float t=zero;
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
                    float *c=C;
                    float *at=A;
                    float *bt=B;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A+j*ldA;
                        float *b=B+j*ldB;
                        for(int i=j;i<n;i++)
                        {
                            float s=zero;
                            float t=zero;
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
            cblas_xerbla(8,"cblas_ssyr2k","");
        else if(ldB<(tran==CblasNoTrans?k:n))
            cblas_xerbla(10,"cblas_ssyr2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_ssyr2k","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_ssyr2k(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_ssyr2k","");
    }
}
