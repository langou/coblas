//
//  cblas_drotmg.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

void cblas_drotmg(double *d1, double *d2, double *x1, double y1, double *param)
{
    const double zero=0.0;
    const double one=1.0;
    const double two=2.0;
    const double gam=4096.0;
    const double gamsq=16777216.0;
    const double rgamsq=5.9604645e-8;
    double r=zero;
    double u=zero;
    double p1=zero;
    double p2=zero;
    double q1=zero;
    double q2=zero;
    double h11=zero;
    double h12=zero;
    double h21=zero;
    double h22=zero;
    double flag=zero;
    double temp=zero;
    if((*d1)<zero)
    {
        flag=-one;
        h11=zero;
        h12=zero;
        h21=zero;
        h22=zero;
        *d1=zero;
        *d2=zero;
        *x1=zero;
    }
    else
    {
        p2=(*d2)*y1;
        if(p2==zero)
        {
            flag=-two;
            param[0]=flag;
            return;
        }
        p1=(*d1)*(*x1);
        q2=p2*y1;
        q1=p1*(*x1);
        if(fabs(q1)>fabs(q2))
        {
            h21=-y1/(*x1);
            h12=p2/p1;
            u=one-h12*h21;
            if(u>zero)
            {
                flag=zero;
                *d1/=u;
                *d2/=u;
                *x1*=u;
            }
        }
        else
        {
            if(q2<zero)
            {
                flag=-one;
                h11=zero;
                h12=zero;
                h21=zero;
                h22=zero;
                *d1=zero;
                *d2=zero;
                *x1=zero;
            }
            else
            {
                flag=one;
                h11=p1/p2;
                h22=(*x1)/y1;
                u=one+h11*h22;
                temp=(*d2)/u;
                *d2=(*d1)/u;
                *d1=temp;
                *x1=y1*u;
            }
        }
        if((*d1)!=zero)
        {
            while(((*d1)<=rgamsq)||((*d1)>=gamsq))
            {
                if(flag==zero)
                {
                    h11=one;
                    h22=one;
                    flag=-one;
                }
                else
                {
                    h21=-one;
                    h12=one;
                    flag=-one;
                }
                if((*d1)<=rgamsq)
                {
                    r=gam;
                    *d1*=r*r;
                    *x1/=gam;
                    h11/=gam;
                    h12/=gam;
                }
                else
                {
                    r=gam;
                    *d1/=r*r;
                    *x1*=gam;
                    h11*=gam;
                    h12*=gam;
                }
            }
        }
        if((*d2)!=zero)
        {
            while((fabs(*d2)<=rgamsq)||(fabs(*d2)>=gamsq))
            {
                if(flag==zero)
                {
                    h11=one;
                    h22=one;
                    flag=-one;
                }
                else
                {
                    h21=-one;
                    h12=one;
                    flag=-one;
                }
                if(fabs(*d2)<=rgamsq)
                {
                    r=gam;
                    *d2*=r*r;
                    h21/=gam;
                    h22/=gam;
                }
                else
                {
                    r=gam;
                    *d2/=r*r;
                    h21*=gam;
                    h22*=gam;
                }
            }
        }
    }
    if(flag<zero)
    {
        param[1]=h11;
        param[2]=h21;
        param[3]=h12;
        param[4]=h22;
    } 
    else if(flag==zero) 
    {
        param[2]=h21;
        param[3]=h12;
    } 
    else 
    {
        param[1]=h11;
        param[4]=h22;
    }
    param[0]=flag;
}
