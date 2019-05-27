//
//  cblas_dsymm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dsymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC)
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
            const double zero=0.0;
            if(alpha==zero)
            {
                double *c=C;
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
                    double *b=B;
                    double *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A;
                        for(int i=0;i<m;i++)
                        {
                            double t=alpha*b[i];
                            double s=zero;
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
                    double *b=B;
                    double *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double *a=A+m*ldA;
                        for(int i=m-1;i>=0;i--)
                        {
                            a-=ldA;
                            double t=alpha*b[i];
                            double s=zero;
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
                    double *a=A;
                    double *c=C;
                    double *b=B;
                    for(int j=0;j<n;j++)
                    {
                        double t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        double *at=A+(j+1)*ldA;
                        double *bt=B;
                        for(int k=0;k<j;k++)
                        {
                            double t=alpha*a[k];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            double t=alpha*at[j];
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
                    double *a=A;
                    double *c=C;
                    double *b=B;
                    for(int j=0;j<n;j++)
                    {
                        double t=alpha*a[j];
                        for(int i=0;i<m;i++)
                            c[i]=c[i]*beta+t*b[i];
                        double *bt=B;
                        double *at=A;
                        for(int k=0;k<j;k++)
                        {
                            double t=alpha*at[j];
                            for(int i=0;i<m;i++)
                                c[i]+=bt[i]*t;
                            at+=ldA;
                            bt+=ldB;
                        }
                        bt=B+(j+1)*ldB;
                        for(int k=j+1;k<n;k++)
                        {
                            double t=alpha*at[k];
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
            cblas_xerbla(8,"cblas_dsymm","");
        else if(ldB<n)
            cblas_xerbla(10,"cblas_dsymm","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_dsymm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_dsymm(CblasColMajor,Side,Uplo,n,m,alpha,A,ldA,B,ldB,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_dsymm","");
    }
}
