//
//  cblas_cgemm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_cgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, float complex *Alpha, float complex *A, int ldA, float complex *B, int ldB, float complex *Beta, float complex *C, int ldC)
{
    if((transA!=CblasNoTrans)&&(transA!=CblasTrans)&&(transA!=CblasConjTrans))
        cblas_xerbla(2,"cblas_cgemm","");
    else if((transB!=CblasNoTrans)&&(transB!=CblasTrans)&&(transB!=CblasConjTrans))
        cblas_xerbla(3,"cblas_cgemm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_cgemm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_cgemm","");
    else if(k<0)
        cblas_xerbla(6,"cblas_cgemm","");
    else if(order==CblasColMajor)
    {
        if(ldA<((transA!=CblasNoTrans)?k:m))
            cblas_xerbla(9,"cblas_cgemm","");
        else if(ldB<((transB!=CblasNoTrans)?n:k))
            cblas_xerbla(11,"cblas_cgemm","");
        else if(ldC<m)
            cblas_xerbla(14,"cblas_cgemm","");
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
            else if((transA==CblasNoTrans)&&(transB==CblasNoTrans))
            {
                float complex *c=C;
                float complex *b=B;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    float complex *a=A;
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
            else if((transA==CblasTrans)&&(transB==CblasNoTrans))
            {
                float complex *b=B;
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
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
            else if((transA==CblasConjTrans)&&(transB==CblasNoTrans))
            {
                float complex *b=B;
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
                        for(int l=0;l<k;l++)
                        {
                            s+=conjf(a[l])*b[l];
                        }
                        c[i]=alpha*s+beta*c[i];
                        a+=ldA;
                    }
                    b+=ldB;
                    c+=ldC;
                }
            }
            else if((transA==CblasNoTrans)&&(transB==CblasTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    float complex *a=A;
                    float complex *b=B;
                    for(int l=0;l<k;l++)
                    {
                        float complex t=alpha*b[j];
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
            else if((transA==CblasNoTrans)&&(transB==CblasConjTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    float complex *a=A;
                    float complex *b=B;
                    for(int l=0;l<k;l++)
                    {
                        float complex t=alpha*conjf(b[j]);
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
            else if((transA==CblasTrans)&&(transB==CblasTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
                        float complex *b=B;
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
            else if((transA==CblasConjTrans)&&(transB==CblasTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
                        float complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=conjf(a[l])*b[j];
                            b+=ldB;
                        }
                        c[i]=alpha*s+beta*c[i];
                        a+=ldA;
                    }
                    c+=ldC;
                }
            }
            else if((transA==CblasTrans)&&(transB==CblasConjTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
                        float complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=a[l]*conjf(b[j]);
                            b+=ldB;
                        }
                        c[i]=alpha*s+beta*c[i];
                        a+=ldA;
                    }
                    c+=ldC;
                }
            }
            else if((transA==CblasConjTrans)&&(transB==CblasConjTrans))
            {
                float complex *c=C;
                for(int j=0;j<n;j++)
                {
                    float complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        float complex s=zero;
                        float complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=conjf(a[l])*conjf(b[j]);
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
            cblas_xerbla(9,"cblas_cgemm","");
        else if(ldB<((transB!=CblasNoTrans)?k:n))
            cblas_xerbla(11,"cblas_cgemm","");
        else if(ldC<n)
            cblas_xerbla(14,"cblas_cgemm","");
        else
            cblas_cgemm(CblasColMajor,transB,transA,n,m,k,Alpha,B,ldB,A,ldA,Beta,C,ldC);
    }
    else
    {
        cblas_xerbla(1,"cblas_cgemm","");
    }
}
