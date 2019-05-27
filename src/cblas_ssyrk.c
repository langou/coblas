//
//  cblas_ssyrk.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_ssyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, float alpha, float *A, int ldA, float beta, float *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_ssyrk","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_ssyrk","");
    else if(n<0)
        cblas_xerbla(4,"cblas_ssyrk","");
    else if(k<0)
        cblas_xerbla(5,"cblas_ssyrk","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_ssyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_ssyrk","");
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
                        for(int l=0;l<k;l++)
                        {
                            float t=alpha*a[j];
                            for(int i=0;i<=j;i++)
                                c[i]+=t*a[i];
                            a+=ldA;
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
                        for(int l=0;l<k;l++)
                        {
                            float t=alpha*a[j];
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
                    float *c=C;
                    float *at=A;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A;
                        for(int i=0;i<=j;i++)
                        {
                            float t=zero;
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
                    float *at=A;
                    float *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A+j*ldA;
                        for(int i=j;i<n;i++)
                        {
                            float t=zero;
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
            cblas_xerbla(8,"cblas_ssyrk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_ssyrk","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_ssyrk(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_ssyrk","");
    }
}
