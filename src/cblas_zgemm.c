//
//  cblas_zgemm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_zgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, double complex *Alpha, double complex *A, int ldA, double complex *B, int ldB, double complex *Beta, double complex *C, int ldC)
{
    if((transA!=CblasNoTrans)&&(transA!=CblasTrans)&&(transA!=CblasConjTrans))
        cblas_xerbla(2,"cblas_zgemm","");
    else if((transB!=CblasNoTrans)&&(transB!=CblasTrans)&&(transB!=CblasConjTrans))
        cblas_xerbla(3,"cblas_zgemm","");
    else if(m<0)
        cblas_xerbla(4,"cblas_zgemm","");
    else if(n<0)
        cblas_xerbla(5,"cblas_zgemm","");
    else if(k<0)
        cblas_xerbla(6,"cblas_zgemm","");
    else if(order==CblasColMajor)
    {
        if(ldA<((transA!=CblasNoTrans)?k:m))
            cblas_xerbla(9,"cblas_zgemm","");
        else if(ldB<((transB!=CblasNoTrans)?n:k))
            cblas_xerbla(11,"cblas_zgemm","");
        else if(ldC<m)
            cblas_xerbla(14,"cblas_zgemm","");
        else
        {
            const double complex alpha=*Alpha;
            const double complex beta=*Beta;
            const double complex zero=0.0;
            if(alpha==zero)
            {
                double complex *c=C;
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
                double complex *c=C;
                double complex *b=B;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    double complex *a=A;
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
                double complex *b=B;
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
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
                double complex *b=B;
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
                        for(int l=0;l<k;l++)
                        {
                            s+=conj(a[l])*b[l];
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    double complex *a=A;
                    double complex *b=B;
                    for(int l=0;l<k;l++)
                    {
                        double complex t=alpha*b[j];
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    for(int i=0;i<m;i++)
                        c[i]*=beta;
                    double complex *a=A;
                    double complex *b=B;
                    for(int l=0;l<k;l++)
                    {
                        double complex t=alpha*conj(b[j]);
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
                        double complex *b=B;
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=conj(a[l])*b[j];
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=a[l]*conj(b[j]);
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
                double complex *c=C;
                for(int j=0;j<n;j++)
                {
                    double complex *a=A;
                    for(int i=0;i<m;i++)
                    {
                        double complex s=zero;
                        double complex *b=B;
                        for(int l=0;l<k;l++)
                        {
                            s+=conj(a[l])*conj(b[j]);
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
            cblas_xerbla(9,"cblas_zgemm","");
        else if(ldB<((transB!=CblasNoTrans)?k:n))
            cblas_xerbla(11,"cblas_zgemm","");
        else if(ldC<n)
            cblas_xerbla(14,"cblas_zgemm","");
        else
            cblas_zgemm(CblasColMajor,transB,transA,n,m,k,Alpha,B,ldB,A,ldA,Beta,C,ldC);
    }
    else
    {
        cblas_xerbla(1,"cblas_zgemm","");
    }
}
