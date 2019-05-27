//
//  cblas_zher2k.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zher2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, int n, int k, double complex *Alpha, double complex *A, int ldA, double complex *B, int ldB, double beta, double complex *C, int ldC)
{
    if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(2,"cblas_zher2k","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(3,"cblas_zher2k","");
    else if(n<0)
        cblas_xerbla(4,"cblas_zher2k","");
    else if(k<0)
        cblas_xerbla(5,"cblas_zher2k","");
    else if(order==CblasColMajor)
    {
        if(ldA<(tran==CblasNoTrans?n:k))
            cblas_xerbla(8,"cblas_zher2k","");
        else if(ldB<(tran==CblasNoTrans?n:k))
            cblas_xerbla(10,"cblas_zher2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_zher2k","");
        else
        {
            const double complex alpha=*Alpha;
            const double complex zero=0.0;
            const double rzero=0.0;
            const double one=1.0;
            if(((alpha==zero)||(k==0))&&(beta==one))
                return;
            if(alpha==zero)
            {
                double complex *c=C;
                if(uplo==CblasUpper)
                {
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
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double complex s=alpha*conj(b[j]);
                            double complex t=conj(alpha*a[j]);
                            for(int i=0;i<j;i++)
                                c[i]+=s*a[i]+t*b[i];
                            c[j]=creal(c[j])+creal(a[j]*s+b[j]*t);
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
                        c[j]=beta*creal(c[j]);
                        for(int i=j+1;i<n;i++)
                            c[i]*=beta;
                        double complex *a=A;
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            double complex s=alpha*conj(b[j]);
                            double complex t=conj(alpha*a[j]);
                            c[j]=creal(c[j])+creal(s*a[j]+t*b[j]);
                            for(int i=j+1;i<n;i++)
                                c[i]+=s*a[i]+t*b[i];
                            a+=ldA;
                            b+=ldB;
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
                                s+=conj(b[l])*at[l];
                                t+=conj(a[l])*bt[l];
                            }
                            if(i<j)
                                c[i]=alpha*t+conj(alpha)*s+beta*c[i];
                            else
                                c[j]=creal(alpha*t+conj(alpha)*s)+beta*creal(c[j]);
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
                                s+=conj(b[l])*at[l];
                                t+=conj(a[l])*bt[l];
                            }
                            if(i>j)
                                c[i]=alpha*t+conj(alpha)*s+beta*c[i];
                            else
                                c[j]=creal(alpha*t+conj(alpha)*s)+beta*creal(c[j]);
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
            cblas_xerbla(8,"cblas_zher2k","");
        else if(ldB<(tran==CblasNoTrans?k:n))
            cblas_xerbla(10,"cblas_zher2k","");
        else if(ldC<n)
            cblas_xerbla(13,"cblas_zher2k","");
        else
        {
            double complex alpha=conj(*Alpha);
            CBLAS_TRANSPOSE Tran=(tran==CblasNoTrans)?CblasConjTrans:CblasNoTrans;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_zher2k(CblasColMajor,Uplo,Tran,n,k,&alpha,A,ldA,B,ldB,beta,C,ldC);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_zher2k","");
    }
}
