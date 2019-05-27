//
//  cblas_dtrmm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_dtrmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, CBLAS_DIAG diag, int m, int n, double alpha, double *A, int ldA, double *B, int ldB)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_dtrmm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_dtrmm","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(4,"cblas_dtrmm","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(5,"cblas_dtrmm","");
    else if(m<0)
        cblas_xerbla(6,"cblas_dtrmm","");
    else if(n<0)
        cblas_xerbla(7,"cblas_dtrmm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_dtrmm","");
        else if(ldB<m)
            cblas_xerbla(12,"cblas_dtrmm","");
        else
        {
            const double zero=0.0;
            const bool nounit=(diag==CblasNonUnit);
            if(alpha==zero)
            {
                double *b=B;
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
                        double *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double *a=A;
                            for(int k=0;k<m;k++)
                            {
                                double t=alpha*b[k];
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
                        double *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double *a=A+m*ldA;
                            for(int k=m-1;k>=0;k--)
                            {
                                a-=ldA;
                                double t=alpha*b[k];
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
                else if((tran==CblasTrans)||(tran==CblasConjTrans))
                {
                    if(uplo==CblasUpper)
                    {
                        double *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double *a=A+m*ldA;
                            for(int i=m-1;i>=0;i--)
                            {
                                a-=ldA;
                                double t=b[i];
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
                        double *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double *a=A;
                            for(int i=0;i<m;i++)
                            {
                                double t=b[i];
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
            }
            else if(side==CblasRight)
            {
                if(tran==CblasNoTrans)
                {
                    if(uplo==CblasUpper)
                    {
                        double *a=A+n*ldA;
                        double *b=B+n*ldB;
                        for(int j=n-1;j>=0;j--)
                        {
                            a-=ldA;
                            b-=ldB;
                            double s=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            double *bt=B;
                            for(int k=0;k<j;k++)
                            {
                                double t=alpha*a[k];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                bt+=ldB;
                            }
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double *a=A;
                        double *b=B;
                        for(int j=0;j<n;j++)
                        {
                            double s=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            double *bt=b+ldB;
                            for(int k=j+1;k<n;k++)
                            {
                                double t=alpha*a[k];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                bt+=ldB;
                            }
                            a+=ldA;
                            b+=ldB;
                        }
                    }
                }
                else if((tran==CblasTrans)||(tran==CblasConjTrans))
                {
                    if(uplo==CblasUpper)
                    {
                        double *a=A;
                        double *bt=B;
                        for(int k=0;k<n;k++)
                        {
                            double *b=B;
                            for(int j=0;j<k;j++)
                            {
                                double t=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                b+=ldB;
                            }
                            double s=(nounit)?alpha*a[k]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            a+=ldA;
                            bt+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        double *a=A+n*ldA;
                        double *bt=B+n*ldB;
                        for(int k=n-1;k>=0;k--)
                        {
                            a-=ldA;
                            bt-=ldB;
                            double *b=B+(k+1)*ldB;
                            for(int j=k+1;j<n;j++)
                            {
                                double t=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                b+=ldB;
                            }
                            double s=(nounit)?alpha*a[k]:alpha;
                            for(int i=0;i<m;i++)
                                bt[i]*=s;
                        }
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_dtrmm","");
        else if(ldB<n)
            cblas_xerbla(12,"cblas_dtrmm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_dtrmm(CblasColMajor,Side,Uplo,tran,diag,n,m,alpha,A,ldA,B,ldB);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_dtrmm","");
    }
}
