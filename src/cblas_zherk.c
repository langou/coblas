//
//  cblas_zherk.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zherk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double alpha, double complex *A, int ldA, double beta, double complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zherk","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_zherk","");
    else if(n<0)
        cblas_xerbla(4,"cblas_zherk","");
    else if(k<0)
        cblas_xerbla(5,"cblas_zherk","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_zherk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_zherk","");
        else
        {
            const double complex zero=0.0;
            const double rzero=0.0;
            const double one=1.0;
            if(((alpha==rzero)||(k==0))&&(beta==one))
                return;
            if(alpha==rzero)
            {
                if(uplo==CblasUpper)
                {
                    double complex *c=C;
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
                            c[j]=beta*creal(c[j]);
                            c+=ldC;
                        }
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *c=C;
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
                            c[j]=beta*creal(c[j]);
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
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        for(int i=0;i<j;i++)
                            c[i]*=beta;
                        c[j]=beta*creal(c[j]);
                        double complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double complex t=alpha*conj(a[j]);
                            for(int i=0;i<j;i++)
                                c[i]+=t*a[i];
                            c[j]=creal(c[j])+creal(t*a[j]);
                            a+=ldA;
                        }
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        c[j]=beta*creal(c[j]);
                        for(int i=j+1;i<n;i++)
                            c[i]*=beta;
                        double complex *a=A;
                        for(int l=0;l<k;l++)
                        {
                            double complex t=alpha*conj(a[j]);
                            c[j]=creal(c[j])+creal(t*a[j]);
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
                    double complex *c=C;
                    double complex *at=A;
                    for(int j=0;j<n;j++)
                    {
                        double complex *a=A;
                        for(int i=0;i<j;i++)
                        {
                            double complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=conj(a[l])*at[l];
                            c[i]=alpha*t+beta*c[i];
                            a+=ldA;
                        }
                        double s=rzero;
                        for(int l=0;l<k;l++)
                            s+=creal(conj(at[l])*at[l]);
                        c[j]=alpha*s+beta*creal(c[j]);
                        at+=ldA;
                        c+=ldC;
                    }
                }
                else if(uplo==CblasLower)
                {
                    double complex *at=A;
                    double complex *c=C;
                    for(int j=0;j<n;j++)
                    {
                        double s=rzero;
                        for(int l=0;l<k;l++)
                            s+=creal(conj(at[l])*at[l]);
                        c[j]=alpha*s+beta*creal(c[j]);
                        double complex *a=A+(j+1)*ldA;
                        for(int i=j+1;i<n;i++)
                        {
                            double complex t=zero;
                            for(int l=0;l<k;l++)
                                t+=conj(a[l])*at[l];
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
            cblas_xerbla(8,"cblas_zherk","");
        else if(ldC<n)
            cblas_xerbla(11,"cblas_zherk","");
        else
        {
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasConjTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_zherk(CblasColMajor,Uplo,Tran,n,k,alpha,A,ldA,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_zherk","");
    }
}
