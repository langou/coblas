//
//  cblas_strmm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_strmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, CBLAS_DIAG diag, int m, int n, float alpha, float *A, int ldA, float *B, int ldB)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_strmm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_strmm","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(4,"cblas_strmm","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(5,"cblas_strmm","");
    else if(m<0)
        cblas_xerbla(6,"cblas_strmm","");
    else if(n<0)
        cblas_xerbla(7,"cblas_strmm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_strmm","");
        else if(ldB<m)
            cblas_xerbla(12,"cblas_strmm","");
        else
        {
            const float zero=0.0f;
            const bool nounit=(diag==CblasNonUnit);
            if(alpha==zero)
            {
                float *b=B;
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
                        float *b=B;
                        for(int j=0;j<n;j++)
                        {
                            float *a=A;
                            for(int k=0;k<m;k++)
                            {
                                float t=alpha*b[k];
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
                        float *b=B;
                        for(int j=0;j<n;j++)
                        {
                            float *a=A+m*ldA;
                            for(int k=m-1;k>=0;k--)
                            {
                                a-=ldA;
                                float t=alpha*b[k];
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
                        float *b=B;
                        for(int j=0;j<n;j++)
                        {
                            float *a=A+m*ldA;
                            for(int i=m-1;i>=0;i--)
                            {
                                a-=ldA;
                                float t=b[i];
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
                        float *b=B;
                        for(int j=0;j<n;j++)
                        {
                            float *a=A;
                            for(int i=0;i<m;i++)
                            {
                                float t=b[i];
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
                        float *a=A+n*ldA;
                        float *b=B+n*ldB;
                        for(int j=n-1;j>=0;j--)
                        {
                            a-=ldA;
                            b-=ldB;
                            float s=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            float *bt=B;
                            for(int k=0;k<j;k++)
                            {
                                float t=alpha*a[k];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                bt+=ldB;
                            }
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        float *a=A;
                        float *b=B;
                        for(int j=0;j<n;j++)
                        {
                            float s=(nounit)?alpha*a[j]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            float *bt=b+ldB;
                            for(int k=j+1;k<n;k++)
                            {
                                float t=alpha*a[k];
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
                        float *a=A;
                        float *bt=B;
                        for(int k=0;k<n;k++)
                        {
                            float *b=B;
                            for(int j=0;j<k;j++)
                            {
                                float t=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                b+=ldB;
                            }
                            float s=(nounit)?alpha*a[k]:alpha;
                            for(int i=0;i<m;i++)
                                b[i]*=s;
                            a+=ldA;
                            bt+=ldB;
                        }
                    }
                    else if(uplo==CblasLower)
                    {
                        float *a=A+n*ldA;
                        float *bt=B+n*ldB;
                        for(int k=n-1;k>=0;k--)
                        {
                            a-=ldA;
                            bt-=ldB;
                            float *b=B+(k+1)*ldB;
                            for(int j=k+1;j<n;j++)
                            {
                                float t=alpha*a[j];
                                for(int i=0;i<m;i++)
                                    b[i]+=t*bt[i];
                                b+=ldB;
                            }
                            float s=(nounit)?alpha*a[k]:alpha;
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
            cblas_xerbla(10,"cblas_strmm","");
        else if(ldB<n)
            cblas_xerbla(12,"cblas_strmm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_strmm(CblasColMajor,Side,Uplo,tran,diag,n,m,alpha,A,ldA,B,ldB);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_strmm","");
    }
}
