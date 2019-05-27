//
//  cblas_drotm.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>

void cblas_drotm(int n, double *x, int incx, double *y, int incy, double *param)
{
    const int flag=(int)(param[0]);
    const double h11=param[1];
    const double h21=param[2];
    const double h12=param[3];
    const double h22=param[4];
    if((incx==1)&&(incy==1))
    {
        if(flag==-1)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i];
                double z=y[i];
                x[i]=w*h11+z*h12;
                y[i]=w*h21+z*h22;
            }
        }
        else if(flag==0)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i];
                double z=y[i];
                x[i]=w+z*h12;
                y[i]=w*h21+z;
            }
        }
        else if(flag==1)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i];
                double z=y[i];
                x[i]=w*h11+z;
                y[i]=-w+z*h22;
            }
        }
    }
    else
    {
        if(incx<0)
            x-=(n-1)*incx;
        if(incy<0)
            y-=(n-1)*incy;
        if(flag==-1)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i*incx];
                double z=y[i*incy];
                x[i*incx]=w*h11+z*h12;
                y[i*incy]=w*h21+z*h22;
            }
        }
        else if(flag==0)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i*incx];
                double z=y[i*incy];
                x[i*incx]=w+z*h12;
                y[i*incy]=w*h21+z;
            }
        }
        else if(flag==1)
        {
            for(int i=0;i<n;i++)
            {
                double w=x[i*incx];
                double z=y[i*incy];
                x[i*incx]=w*h11+z;
                y[i*incy]=-w+z*h22;
            }
        }
    }
}
