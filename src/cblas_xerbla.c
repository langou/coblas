//
//  cblas_xerbla.c
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#include <coblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>

void cblas_xerbla(int info, char *rout, char *form, ...)
{
    va_list argptr;
    va_start(argptr,form);
    fprintf(stderr,"COBLAS error: Parameter %d passed to %s was invalid.\n",info,rout);
    vfprintf(stderr,form,argptr);
    va_end(argptr);
    exit(EXIT_FAILURE);
}
