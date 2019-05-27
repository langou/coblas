//
//  cblas_srotmg.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <math.h>

void cblas_srotmg(float *d1, float *d2, float *x1, float y1, float *param)
{
    const float zero=0.0f;
    const float one=1.0f;
    const float two=2.0f;
    const float gam=4096.0f;
    const float gamsq=16777216.0f;
    const float rgamsq=5.9604645e-8f;
    float r=zero;
    float u=zero;
    float p1=zero;
    float p2=zero;
    float q1=zero;
    float q2=zero;
    float h11=zero;
    float h12=zero;
    float h21=zero;
    float h22=zero;
    float flag=zero;
    float temp=zero;
    
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
        if(fabsf(q1)>fabsf(q2))
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
                temp=*d2/u;
                *d2=*d1/u;
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
            while((fabsf(*d2)<=rgamsq)||(fabsf(*d2)>=gamsq))
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
                if(fabsf(*d2)<=rgamsq)
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
