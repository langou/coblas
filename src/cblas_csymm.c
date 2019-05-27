//
//  cblas_csymm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_csymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, float complex *Alpha, float complex *A, int ldA, float complex *B, int ldB, float complex *Beta, float complex *C, int ldC)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_csymm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_csymm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_csymm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_csymm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(8,"cblas_csymm","");
        else if(ldB<m)
            cblas_xerbla(10,"cblas_csymm","");
        else if(ldC<m)
            cblas_xerbla(13,"cblas_csymm","");
        else
        {
            const float complex alpha=*Alpha;
            const float complex beta=*Beta;
            const float complex zero=0.0f;
            if(alpha==zero)
            {
                float complex *c=C;
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
                    float complex *b=B;
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float complex *a=A;
                        for(int i=0;i<m;i++)
                        {
                            float complex t=alpha*b[i];
                            float complex s=zero;
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
                    float complex *b=B;
                    float complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        float complex *a=A+m*ldA;
                        for(int i=m-1;i>=0;i--)
                        {
                            a-=ldA;
                            float complex t=alpha*b[i];
                            float complex s=zero;
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
                    float complex *a=A;
                    float complex *c=C;
                    float complex *b=B;
                    for(int j=0;j<n;j++)
                    {
                        float complex t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        float complex *at=A+(j+1)*ldA;
                        float complex *bt=B;
                        for(int k=0;k<j;k++)
                        {
                            float complex t=alpha*a[k];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            float complex t=alpha*at[j];
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
                    float complex *a=A;
                    float complex *c=C;
                    float complex *b=B;
                    for(int j=0;j<n;j++)
                    {
                        float complex t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        float complex *bt=B;
                        float complex *at=A;
                        for(int k=0;k<j;k++)
                        {
                            float complex t=alpha*at[j];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            at+=ldA;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            float complex t=alpha*at[k];
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
            cblas_xerbla(8,"cblas_csymm","");
        else if(ldB<n)
            cblas_xerbla(10,"cblas_csymm","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_csymm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_csymm(CblasColMajor,Side,Uplo,n,m,Alpha,A,ldA,B,ldB,Beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_csymm","");
    }
}
