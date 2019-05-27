//
//  cblas_ssymm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_ssymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_dsymm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_dsymm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_dsymm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_dsymm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(8,"cblas_dsymm","");
        else if(ldB<m)
            cblas_xerbla(10,"cblas_dsymm","");
        else if(ldC<m)
            cblas_xerbla(13,"cblas_dsymm","");
        else
        {
            const float zero=0.0f;
            if(alpha==zero)
            {
                float *c=C;
                if(beta==zero)
                {
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<m;i++)
                            c[i]=zero;
                        c+=ldC;
                    }
                }
                else
                {
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<m;i++)
                            c[i]*=beta;
                        c+=ldC;
                    }
                }
            }
            else if(side==CblasLeft)
            {
                if(uplo==CblasUpper)
                {
                    float *b=B;
                    float *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A;
                        for(int i=0;i<m;i++)
                        {
                            float t=alpha*b[i];
                            float s=zero;
                            for(int k=0;k<i;k++)
                            {
                                c[k]+=t*a[k];
                                s+=b[k]*a[k];
                            }
                            c[i]=beta*c[i]+t*a[i]+alpha*s;
                            a+=ldA;
                        }
                        b+=ldB;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    float *b=B;
                    float *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float *a=A+m*ldA;
                        for(int i=m-1;i>=0;i--)
                        {
                            a-=ldA;
                            float t=alpha*b[i];
                            float s=zero;
                            for(int k=i+1;k<m;k++)
                            {
                                c[k]+=t*a[k];
                                s+=b[k]*a[k];
                            }
                            c[i]=beta*c[i]+t*a[i]+alpha*s;
                        }
                        b+=ldB;
                        c+=ldC;
                    }
                }
            }
            else if(side==CblasRight)
            {
                if(uplo==CblasUpper)
                {
                    float *a=A;
                    float *c=C;
                    float *b=B;
                    for(int j=0;j<n;j++)
                    {
                        float t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        float *at=A+(j+1)*ldA;
                        float *bt=B;
                        for(int k=0;k<j;k++)
                        {
                            float t=alpha*a[k];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            float t=alpha*at[j];
                            for(int i=0;i<m;i++)
                                c[i]+=t*bt[i];
                            at+=ldA;
                            bt+=ldB;
                        }
                        a+=ldA;
                        b+=ldB;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    float *a=A;
                    float *c=C;
                    float *b=B;
                    for(int j=0;j<n;j++)
                    {
                        float t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        float *bt=B;
                        float *at=A;
                        for(int k=0;k<j;k++)
                        {
                            float t=alpha*at[j];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            at+=ldA;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            float t=alpha*at[k];
                            for(int i=0;i<m;i++)
                                c[i]+=t*bt[i];
                            bt+=ldB;
                        }
                        a+=ldA;
                        b+=ldB;
                        c+=ldC;
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(8,"cblas_ssymm","");
        else if(ldB<n)
            cblas_xerbla(10,"cblas_ssymm","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_ssymm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_ssymm(CblasColMajor,Side,Uplo,n,m,alpha,A,ldA,B,ldB,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_ssymm","");
    }
}
