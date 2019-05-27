//
//  cblas_dgemm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC)
{
    if((transA!=CblasNoTrans)&&(transA!=CblasTrans)&&(transA!=CblasConjTrans))
        cblas_xerbla(2,"cblas_dgemm","");
    else if((transB!=CblasNoTrans)&&(transB!=CblasTrans)&&(transB!=CblasConjTrans))
        cblas_xerbla(3,"cblas_dgemm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_dgemm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_dgemm","");
    else if(k<0)
        cblas_xerbla(6,"cblas_dgemm","");
    else if(order==CblasColMajor)
    {
        if(ldA<((transA!=CblasNoTrans)?k:m))
            cblas_xerbla(9,"cblas_dgemm","");
        else if(ldB<((transB!=CblasNoTrans)?n:k))
            cblas_xerbla(11,"cblas_dgemm","");
        else if(ldC<m)
            cblas_xerbla(14,"cblas_dgemm","");
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
            else if((transA==CblasNoTrans)&&(transB==CblasNoTrans))
            {
                double *c=C;
                double *b=B;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    double *a=A;
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
                double *b=B;
                double *c=C;
                for(int j=0;j<n;j++)
                {
                    double *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double s=zero;
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
                double *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    double *a=A;
                    double *b=B;
                    for(int l=0;l<k;l++)
                    {
                        double t=alpha*b[j];
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
                double *c=C;
                for(int j=0;j<n;j++)
                {
                    double *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double s=zero;
                        double *b=B;
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
            cblas_xerbla(9,"cblas_dgemm","");
        else if(ldB<((transB!=CblasNoTrans)?k:n))
            cblas_xerbla(11,"cblas_dgemm","");
        else if(ldC<n)
            cblas_xerbla(14,"cblas_dgemm","");
        else
            cblas_dgemm(CblasColMajor,transB,transA,n,m,k,alpha,B,ldB,A,ldA,beta,C,ldC);
    }
    else
    {
        cblas_xerbla(1,"cblas_dgemm","");
    }
}
