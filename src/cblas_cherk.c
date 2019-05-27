//
//  cblas_cherk.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_cherk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, float alpha, float complex *A, int ldA, float beta, float complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_cherk","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_cherk","");
    else if(n<0)
        cblas_xerbla(4,"cblas_cherk","");
    else if(k<0)
        cblas_xerbla(5,"cblas_cherk","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_cherk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_cherk","");
        else
        {
            const float complex zero=0.0f;
            const float rzero=0.0f;
            const float one=1.0f;
            if(((alpha==rzero)||(k==0))&&(beta==one))
                return;
            if(alpha==rzero)
            {
                if(uplo==CblasUpper)
                {
                    float complex *c=C;
                    if(beta==rzero)
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
                            for(int i=0;i<j;i++)
                                c[i]*=beta;
                            c[j]=beta*crealf(c[j]);
                            c+=ldC;
                        }
                    }
                }
                else if(uplo==CblasLower)
                {
                    float complex *c=C;
                    if(beta==rzero)
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
                            c[j]=beta*crealf(c[j]);
                            for(int i=j+1;i<n;i++)
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
                        for(int i=0;i<j;i++)
                            c[i]*=beta;
                        c[j]=beta*crealf(c[j]);
                        float complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            float complex t=alpha*conjf(a[j]);
                            for(int i=0;i<j;i++)
                                c[i]+=t*a[i];
                            c[j]=crealf(c[j])+crealf(t*a[j]);
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        c[j]=beta*crealf(c[j]);
                        for(int i=j+1;i<n;i++)
                            c[i]*=beta;
                        float complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            float complex t=alpha*conjf(a[j]);
                            c[j]=crealf(c[j])+crealf(t*a[j]);
                            for(int i=j+1;i<n;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
            }
            else if(tran==CblasConjTrans)
            {
                if(uplo==CblasUpper)
                {
                    float complex *c=C;
                    float complex *at=A;
                    for(int j=0;j<n;j++)
                    {
                        float complex *a=A;
                        for(int i=0;i<j;i++)
                        {
                            float complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=conjf(a[l])*at[l];
                            c[i]=alpha*t+beta*c[i];
                            a+=ldA;
                        }
                        float s=rzero;
                        for(int l=0;l<k;l++)
                            s+=crealf(conjf(at[l])*at[l]);
                        c[j]=alpha*s+beta*crealf(c[j]);
                        at+=ldA;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    float complex *at=A;
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float s=rzero;
                        for(int l=0;l<k;l++)
                            s+=crealf(conjf(at[l])*at[l]);
                        c[j]=alpha*s+beta*crealf(c[j]);
                        float complex *a=A+(j+1)*ldA;
                        for(int i=j+1;i<n;i++)
                        {
                            float complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=conjf(a[l])*at[l];
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
            cblas_xerbla(8,"cblas_cherk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_cherk","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasConjTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_cherk(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_cherk","");
    }
}
