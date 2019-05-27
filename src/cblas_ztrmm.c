//
//  cblas_ztrmm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_ztrmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, CBLAS_DIAG diag, int m, int n, double complex *Alpha, double complex *A, int ldA, double complex *B, int ldB)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_ztrmm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_ztrmm","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(4,"cblas_ztrmm","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(5,"cblas_ztrmm","");
    else if(m<0)
        cblas_xerbla(6,"cblas_ztrmm","");
    else if(n<0)
        cblas_xerbla(7,"cblas_ztrmm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_ztrmm","");
        else if(ldB<m)
            cblas_xerbla(12,"cblas_ztrmm","");
        else
        {
            const double complex alpha=*Alpha;
            const double complex zero=0.0;
            const bool nounit=(diag==CblasNonUnit);
            if(alpha==zero)
            {
                double complex *b=B;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        b[i]=zero;
                    b+=ldB;
                }
            }
            else if(side==CblasLeft)
            {
                if(tran==CblasNoTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A;
                            for(int k=0;k<m;k++)
                            {
                                double complex t=alpha*b[k];
                                for(int i=0;i<k;i++)
                                    b[i]+=t*a[i];
                                if(nounit)
                                    t*=a[k];
                                b[k]=t;
                                a+=ldA;
                            }
                            b+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A+m*ldA;
                            for(int k=m-1;k>=0;k--)
                            {
                                a-=ldA;
                                double complex t=alpha*b[k];
                                b[k]=t;
                                if(nounit)
                                    b[k]*=a[k];
                                for(int i=k+1;i<m;i++)
                                    b[i]+=t*a[i];
                            }
                            b+=ldB;
                        }
                    }
                }
                else if(tran==CblasTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A+m*ldA;
                            for(int i=m-1;i>=0;i--)
                            {
                                a-=ldA;
                                double complex t=b[i];
                                if(nounit)
                                    t*=a[i];
                                for(int k=0;k<i;k++)
                                    t+=a[k]*b[k];
                                b[i]=alpha*t;
                            }
                            b+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A;
                            for(int i=0;i<m;i++)
                            {
                                double complex t=b[i];
                                if(nounit)
                                    t*=a[i];
                                for(int k=i+1;k<m;k++)
                                    t+=a[k]*b[k];
                                b[i]=alpha*t;
                                a+=ldA;
                            }
                            b+=ldB;
                        }
                    }
                }
                else if(tran==CblasConjTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A+m*ldA;
                            for(int i=m-1;i>=0;i--)
                            {
                                a-=ldA;
                                double complex t=b[i];
                                if(nounit)
                                    t*=conj(a[i]);
                                for(int k=0;k<i;k++)
                                    t+=conj(a[k])*b[k];
                                b[i]=alpha*t;
                            }
                            b+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex *a=A;
                            for(int i=0;i<m;i++)
                            {
                                double complex t=b[i];
                                if(nounit)
                                    t*=conj(a[i]);
                                for(int k=i+1;k<m;k++)
                                    t+=conj(a[k])*b[k];
                                b[i]=alpha*t;
                                a+=ldA;
                            }
                            b+=ldB;
                        }
                    }
                }
            }
            else if(side==CblasRight)
            {
                if(tran==CblasNoTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *a=A+n*ldA;
                        double complex *b=B+n*ldB;
                        for(int j=n-1;j>=0;j--)
                        {
                            a-=ldA;
                            b-=ldB;
                            double complex t=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=t;
                            double complex *bt=B;
                            for(int k=0;k<j;k++)
                            {
                                double complex s=alpha*a[k];
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                bt+=ldB;
                            }
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *a=A;
                        double complex *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double complex t=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=t;
                            double complex *bt=b+ldB;
                            for(int k=j+1;k<n;k++)
                            {
                                double complex s=alpha*a[k];
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                bt+=ldB;
                            }
                            a+=ldA;
                            b+=ldB;
                        }
                    }
                }
                else if(tran==CblasTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *a=A;
                        double complex *bt=B;
                        for(int k=0;k<n;k++)
                        {
                            double complex *b=B;
                            for(int j=0;j<k;j++)
                            {
                                double complex s=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                b+=ldB;
                            }
                            double complex t=(nounit)?alpha*a[k]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=t;
                            a+=ldA;
                            bt+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *a=A+n*ldA;
                        double complex *bt=B+n*ldB;
                        for(int k=n-1;k>=0;k--)
                        {
                            a-=ldA;
                            bt-=ldB;
                            double complex *b=B+(k+1)*ldB;
                            for(int j=k+1;j<n;j++)
                            {
                                double complex s=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                b+=ldB;
                            }
                            double complex t=(nounit)?alpha*a[k]:alpha;
                            for(int i=0;i<m;i++)
                                bt[i]*=t;
                        }
                    }
                }
                else if(tran==CblasConjTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double complex *a=A;
                        double complex *bt=B;
                        for(int k=0;k<n;k++)
                        {
                            double complex *b=B;
                            for(int j=0;j<k;j++)
                            {
                                double complex s=alpha*conj(a[j]);
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                b+=ldB;
                            }
                            double complex t=(nounit)?alpha*conj(a[k]):alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=t;
                            a+=ldA;
                            bt+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double complex *a=A+n*ldA;
                        double complex *bt=B+n*ldB;
                        for(int k=n-1;k>=0;k--)
                        {
                            a-=ldA;
                            bt-=ldB;
                            double complex *b=B+(k+1)*ldB;
                            for(int j=k+1;j<n;j++)
                            {
                                double complex s=alpha*conj(a[j]);
                                for(int i=0;i<m;i++)
                                    b[i]+=s*bt[i];
                                b+=ldB;
                            }
                            double complex t=(nounit)?alpha*conj(a[k]):alpha;
                            for(int i=0;i<m;i++)
                                bt[i]*=t;
                        }
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_ztrmm","");
        else if(ldB<n)
            cblas_xerbla(12,"cblas_ztrmm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_ztrmm(CblasColMajor,Side,Uplo,tran,diag,n,m,Alpha,A,ldA,B,ldB);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_ztrmm","");
    }
}
