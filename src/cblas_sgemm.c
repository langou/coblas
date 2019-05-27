//
//  cblas_sgemm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC)
{
    if((transA!=CblasNoTrans)&&(transA!=CblasTrans)&&(transA!=CblasConjTrans))
        cblas_xerbla(2,"cblas_sgemm","");
    else if((transB!=CblasNoTrans)&&(transB!=CblasTrans)&&(transB!=CblasConjTrans))
        cblas_xerbla(3,"cblas_sgemm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_sgemm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_sgemm","");
    else if(k<0)
        cblas_xerbla(6,"cblas_sgemm","");
    else if(order==CblasColMajor)
    {
        if(ldA<((transA!=CblasNoTrans)?k:m))
            cblas_xerbla(9,"cblas_sgemm","");
        else if(ldB<((transB!=CblasNoTrans)?n:k))
            cblas_xerbla(11,"cblas_sgemm","");
        else if(ldC<m)
            cblas_xerbla(14,"cblas_sgemm","");
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
            else if((transA==CblasNoTrans)&&(transB==CblasNoTrans))
            {
                float *c=C;
                float *b=B;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    float *a=A;
                    for(int l=0;l<k;l++)
                    {
                        for(int i=0;i<m;i++)
                        {
                            c[i]+=alpha*b[l]*a[i];
                        }
                        a+=ldA;
                    }
                    b+=ldB;
                    c+=ldC;
                }
            }
            else if((transA!=CblasNoTrans)&&(transB==CblasNoTrans))
            {
                float *b=B;
                float *c=C;
                for(int j=0;j<n;j++)
                {
                    float *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float s=zero;
                        for(int l=0;l<k;l++)
                        {
                            s+=a[l]*b[l];
                        }
                        c[i]=alpha*s+beta*c[i];
                        a+=ldA;
                    }
                    b+=ldB;
                    c+=ldC;
                }
            }
            else if((transA==CblasNoTrans)&&(transB!=CblasNoTrans))
            {
                float *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    float *a=A;
                    float *b=B;
                    for(int l=0;l<k;l++)
                    {
                        float t=alpha*b[j];
                        for(int i=0;i<m;i++)
                        {
                            c[i]+=t*a[i];
                        }
                        a+=ldA;
                        b+=ldB;
                    }
                    c+=ldC;
                }
            }
            else if((transA!=CblasNoTrans)&&(transB!=CblasNoTrans))
            {
                float *c=C;
                for(int j=0;j<n;j++)
                {
                    float *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float s=zero;
                        float *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=a[l]*b[j];
                            b+=ldB;
                        }
                        c[i]=alpha*s+beta*c[i];
                        a+=ldA;
                    }
                    c+=ldC;
                }
            }
        }
    }
    else if(order==CblasRowMajor)
    {
        if(ldA<((transA!=CblasNoTrans)?m:k))
            cblas_xerbla(9,"cblas_sgemm","");
        else if(ldB<((transB!=CblasNoTrans)?k:n))
            cblas_xerbla(11,"cblas_sgemm","");
        else if(ldC<n)
            cblas_xerbla(14,"cblas_sgemm","");
        else
            cblas_sgemm(CblasColMajor,transB,transA,n,m,k,alpha,B,ldB,A,ldA,beta,C,ldC);
    }
    else
    {
        cblas_xerbla(1,"cblas_sgemm","");
    }
}
