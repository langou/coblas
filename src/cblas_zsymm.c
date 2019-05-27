//
//  cblas_zsymm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zsymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, double complex *Alpha, double complex *A, int ldA, double complex *B, int ldB, double complex *Beta, double complex *C, int ldC)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_zsymm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_zsymm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_zsymm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_zsymm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(8,"cblas_zsymm","");
        else if(ldB<m)
            cblas_xerbla(10,"cblas_zsymm","");
        else if(ldC<m)
            cblas_xerbla(13,"cblas_zsymm","");
        else
        {
            const double complex alpha=*Alpha;
            const double complex beta=*Beta;
            const double complex zero=0.0;
            if(alpha==zero)
            {
                double complex *c=C;
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
                    double complex *b=B;
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A;
                        for(int i=0;i<m;i++)
                        {
                            double complex t=alpha*b[i];
                            double complex s=zero;
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
                    double complex *b=B;
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A+m*ldA;
                        for(int i=m-1;i>=0;i--)
                        {
                            a-=ldA;
                            double complex t=alpha*b[i];
                            double complex s=zero;
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
                    double complex *a=A;
                    double complex *c=C;
                    double complex *b=B;
                    for(int j=0;j<n;j++)
                    {
                        double complex t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        double complex *bt=B;
                        for(int k=0;k<j;k++)
                        {
                            double complex t=alpha*a[k];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            bt+=ldB;
                        }
                        double complex *at=A+(j+1)*ldA;
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            double complex t=alpha*at[j];
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
                    double complex *a=A;
                    double complex *c=C;
                    double complex *b=B;
                    for(int j=0;j<n;j++)
                    {
                        double complex t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        double complex *bt=B;
                        double complex *at=A;
                        for(int k=0;k<j;k++)
                        {
                            double complex t=alpha*at[j];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            at+=ldA;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            double complex t=alpha*at[k];
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
            cblas_xerbla(8,"cblas_zsymm","");
        else if(ldB<n)
            cblas_xerbla(10,"cblas_zsymm","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_zsymm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_zsymm(CblasColMajor,Side,Uplo,n,m,Alpha,A,ldA,B,ldB,Beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_zsymm","");
    }
    
}
