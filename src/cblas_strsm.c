//
//  cblas_strsm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdbool.h>

void cblas_strsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE tran, CBLAS_DIAG diag, int m, int n, float alpha, float *A, int ldA, float *B, int ldB)
{
    if((side!=CblasLeft)&&(side!=CblasRight))
        cblas_xerbla(2,"cblas_strsm","");
    else if((uplo!=CblasUpper)&&(uplo!=CblasLower))
        cblas_xerbla(3,"cblas_strsm","");
    else if((tran!=CblasNoTrans)&&(tran!=CblasTrans)&&(tran!=CblasConjTrans))
        cblas_xerbla(4,"cblas_strsm","");
    else if((diag!=CblasUnit)&&(diag!=CblasNonUnit))
        cblas_xerbla(5,"cblas_strsm","");
    else if(m<0)
        cblas_xerbla(6,"cblas_strsm","");
    else if(n<0)
        cblas_xerbla(7,"cblas_strsm","");
    else if(order==CblasColMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_strsm","");
        else if(ldB<m)
            cblas_xerbla(12,"cblas_strsm","");
        else
        {
            const float zero=0.0f;
            const float one=1.0f;
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
            else
            {
                if(side==CblasLeft)
                {
                    if(tran==CblasNoTrans)
                    {
                        if(uplo==CblasUpper)
                        {
                            float *b=B;
                            for(int j=0;j<n;j++)
                            {
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                                float *a=A+m*ldA;
                                for(int k=m-1;k>=0;k--)
                                {
                                    a-=ldA;
                                    if(nounit)
                                        b[k]=b[k]/a[k];
                                    for(int i=0;i<k;i++)
                                        b[i]-=b[k]*a[i];
                                }
                                b+=ldB;
                            }
                        }
                        else if(uplo==CblasLower)
                        {
                            float *b=B;
                            for(int j=0;j<n;j++)
                            {
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                                float *a=A;
                                for(int k=0;k<m;k++)
                                {
                                    if(nounit)
                                        b[k]=b[k]/a[k];
                                    for(int i=k+1;i<m;i++)
                                        b[i]-=b[k]*a[i];
                                    a+=ldA;
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
                                float *a=A;
                                for(int i=0;i<m;i++)
                                {
                                    float t=alpha*b[i];
                                    for(int k=0;k<i;k++)
                                        t-=a[k]*b[k];
                                    if(nounit)
                                        t=t/a[i];
                                    b[i]=t;
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
                                for(int i=m-1;i>=0;i--)
                                {
                                    a-=ldA;
                                    float t=alpha*b[i];
                                    for(int k=i+1;k<m;k++)
                                        t-=a[k]*b[k];
                                    if(nounit)
                                        t=t/a[i];
                                    b[i]=t;
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
                            float *b=B;
                            float *a=A;
                            for(int j=0;j<n;j++)
                            {
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                                float *bt=B;
                                for(int k=0;k<j;k++)
                                {
                                    for(int i=0;i<m;i++)
                                        b[i]-=a[k]*bt[i];
                                    bt+=ldB;
                                }
                                if(nounit)
                                {
                                    float t=one/a[j];
                                    for(int i=0;i<m;i++)
                                        b[i]*=t;
                                }
                                a+=ldA;
                                b+=ldB;
                            }
                        }
                        else if(uplo==CblasLower)
                        {
                            float *b=B+n*ldB;
                            float *a=A+n*ldA;
                            for(int j=n-1;j>=0;j--)
                            {
                                a-=ldA;
                                b-=ldB;
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                                float *bt=B+(j+1)*ldB;
                                for(int k=j+1;k<n;k++)
                                {
                                    for(int i=0;i<m;i++)
                                        b[i]-=a[k]*bt[i];
                                    bt+=ldB;
                                }
                                if(nounit)
                                {
                                    float t=one/a[j];
                                    for(int i=0;i<m;i++)
                                        b[i]*=t;
                                }
                            }
                        }
                    }
                    else if((tran==CblasTrans)||(tran==CblasConjTrans))
                    {
                        if(uplo==CblasUpper)
                        {
                            float *b=B+n*ldB;
                            float *a=A+n*ldA;
                            for(int k=n-1;k>=0;k--)
                            {
                                a-=ldA;
                                b-=ldB;
                                if(nounit)
                                {
                                    float t=one/a[k];
                                    for(int i=0;i<m;i++)
                                        b[i]*=t;
                                }
                                float *bt=B;
                                for(int j=0;j<k;j++)
                                {
                                    float t=a[j];
                                    for(int i=0;i<m;i++)
                                        bt[i]-=t*b[i];
                                    bt+=ldB;
                                }
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                            }
                        }
                        else if(uplo==CblasLower)
                        {
                            float *a=A;
                            float *b=B;
                            for(int k=0;k<n;k++)
                            {
                                if(nounit)
                                {
                                    float t=one/a[k];
                                    for(int i=0;i<m;i++)
                                        b[i]*=t;
                                }
                                float *bt=B+(k+1)*ldB;
                                for(int j=k+1;j<n;j++)
                                {
                                    float t=a[j];
                                    for(int i=0;i<m;i++)
                                        bt[i]-=t*b[i];
                                    bt+=ldB;
                                }
                                for(int i=0;i<m;i++)
                                    b[i]*=alpha;
                                a+=ldA;
                                b+=ldB;
                            }
                        }
                    }
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<(side==CblasLeft?m:n))
            cblas_xerbla(10,"cblas_strsm","");
        else if(ldB<n)
            cblas_xerbla(12,"cblas_strsm","");
        else
        {
            CBLAS_SIDE Side=(side==CblasRight)?CblasLeft:CblasRight;
            CBLAS_UPLO Uplo=(uplo==CblasUpper)?CblasLower:CblasUpper;
            cblas_strsm(CblasColMajor,Side,Uplo,tran,diag,n,m,alpha,A,ldA,B,ldB);
        }
    }
    else
    {
        cblas_xerbla(1,"cblas_strsm","");
    }
}
