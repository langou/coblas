//
//  coblas.h
//  COBLAS
//
//  Copyright (c) 2013-2018 University of Colorado Denver. All rights reserved.
//

#ifndef __coblas__
#define __coblas__

typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

#if defined(__cplusplus)
#include <complex>
typedef std::complex<float> CBLAS_CTYPE;
typedef std::complex<double> CBLAS_ZTYPE;
extern "C"
{
#elif (__STDC_VERSION__ >= 199901L)
#include <complex.h>
typedef float complex CBLAS_CTYPE;
typedef double complex CBLAS_ZTYPE;
#else
typedef void CBLAS_CTYPE;
typedef void CBLAS_ZTYPE;
#endif

void cblas_xerbla(int p, char *rout, char *form, ...);

float cblas_sdsdot(int n, float alpha, float *x, int incx, float *y, int incy);
double cblas_dsdot(int n, float *x, int incx, float *y, int incy);
float cblas_sdot(int n, float  *x, int incx, float  *y, int incy);
float cblas_snrm2(int n, float *x, int incx);
float cblas_sasum(int n, float *x, int incx);
int cblas_isamax(int n, float  *x, int incx);
void cblas_sswap(int n, float *x, int incx, float *y, int incy);
void cblas_scopy(int n, float *x, int incx, float *y, int incy);
void cblas_saxpy(int n, float alpha, float *x, int incx, float *y, int incy);
void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_srotmg(float *d1, float *d2, float *b1, float b2, float *P);
void cblas_srot(int n, float *x, int incx, float *y, int incy, float c, float s);
void cblas_srotm(int n, float *x, int incx, float *y, int incy, float *P);
void cblas_sscal(int n, float alpha, float *x, int incx);

double cblas_ddot(int n, double *x, int incx, double *y, int incy);
double cblas_dnrm2(int n, double *x, int incx);
double cblas_dasum(int n, double *x, int incx);
int cblas_idamax(int n, double *x, int incx);
void cblas_dswap(int n, double *x, int incx, double *y, int incy);
void cblas_dcopy(int n, double *x, int incx, double *y, int incy);
void cblas_daxpy(int n, double alpha, double *x, int incx, double *y, int incy);
void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_drotmg(double *d1, double *d2, double *b1, double b2, double *P);
void cblas_drot(int n, double *x, int incx, double *y, int incy, double c, double  s);
void cblas_drotm(int n, double *x, int incx, double *y, int incy, double *P);
void cblas_dscal(int n, double alpha, double *x, int incx);

void cblas_cdotu_sub(int n, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *dotu);
void cblas_cdotc_sub(int n, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *dotc);
float cblas_scnrm2(int n, CBLAS_CTYPE *x, int incx);
float cblas_scasum(int n, CBLAS_CTYPE *x, int incx);
int cblas_icamax(int n, CBLAS_CTYPE *x, int incx);
void cblas_cswap(int n, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy);
void cblas_ccopy(int n, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy);
void cblas_caxpy(int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy);
void cblas_cscal(int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx);
void cblas_csscal(int n, float alpha, CBLAS_CTYPE *x, int incx);

void cblas_zdotu_sub(int n, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *dotu);
void cblas_zdotc_sub(int n, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *dotc);
double cblas_dznrm2(int n, CBLAS_ZTYPE *x, int incx);
double cblas_dzasum(int n, CBLAS_ZTYPE *x, int incx);
int cblas_izamax(int n, CBLAS_ZTYPE *x, int incx);
void cblas_zswap(int n, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy);
void cblas_zcopy(int n, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy);
void cblas_zaxpy(int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy);
void cblas_zscal(int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx);
void cblas_zdscal(int n, double alpha, CBLAS_ZTYPE *x, int incx);

void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy);
void cblas_sgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, int kL, int kU, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy);
void cblas_strmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, float *A, int ldA, float *x, int incx);
void cblas_stbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, float *A, int ldA, float *x, int incx);
void cblas_stpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, float *A, float *x, int incx);
void cblas_strsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, float *A, int ldA, float *x, int incx);
void cblas_stbsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, float *A, int ldA, float *x, int incx);
void cblas_stpsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, float *A, float *x, int incx);
void cblas_ssymv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy);
void cblas_ssbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, float alpha, float *A, int ldA, float *x, int incx, float beta, float *y, int incy);
void cblas_sspmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *A, float *x, int incx, float beta, float *y, int incy);
void cblas_sger(CBLAS_ORDER order, int m, int n, float alpha, float *x, int incx, float *y, int incy, float *A, int ldA);
void cblas_ssyr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *x, int incx, float *A, int ldA);
void cblas_sspr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *x, int incx, float *A);
void cblas_ssyr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *A, int ldA);
void cblas_sspr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *A);

void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, double alpha, double *A, int ldA, double *x, int incx, double beta, double *y, int incy);
void cblas_dgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, int kL, int kU, double alpha, double *A, int ldA, double *x, int incx, double beta, double *y, int incy);
void cblas_dtrmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, double *A, int ldA, double *x, int incx);
void cblas_dtbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, double *A, int ldA, double *x, int incx);
void cblas_dtpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, double *A, double *x, int incx);
void cblas_dtrsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, double *A, int ldA, double *x, int incx);
void cblas_dtbsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, double *A, int ldA, double *x, int incx);
void cblas_dtpsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, double *A, double *x, int incx);
void cblas_dsymv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *A, int ldA, double *x, int incx, double beta, double *y, int incy);
void cblas_dsbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, double alpha, double *A, int ldA, double *x, int incx, double beta, double *y, int incy);
void cblas_dspmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *A, double *x, int incx, double beta, double *y, int incy);
void cblas_dger(CBLAS_ORDER order, int m, int n, double alpha, double *x, int incx, double *y, int incy, double *A, int ldA);
void cblas_dsyr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *A, int ldA);
void cblas_dspr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *A);
void cblas_dsyr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *A, int ldA);
void cblas_dspr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *A);

void cblas_cgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *beta, CBLAS_CTYPE *y, int incy);
void cblas_cgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, int kL, int kU, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *beta, CBLAS_CTYPE *y, int incy);
void cblas_ctrmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx);
void cblas_ctbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx);
void cblas_ctpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_CTYPE *A, CBLAS_CTYPE *x, int incx);
void cblas_ctrsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx);
void cblas_ctbsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx);
void cblas_ctpsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_CTYPE *A, CBLAS_CTYPE *x, int incx);
void cblas_chemv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *beta, CBLAS_CTYPE *y, int incy);
void cblas_chbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *beta, CBLAS_CTYPE *y, int incy);
void cblas_chpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *beta, CBLAS_CTYPE *y, int incy);
void cblas_cgeru(CBLAS_ORDER order, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *A, int ldA);
void cblas_cgerc(CBLAS_ORDER order, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *A, int ldA);
void cblas_cher(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *A, int ldA);
void cblas_chpr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, float alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *A);
void cblas_cher2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *A, int ldA);
void cblas_chpr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *x, int incx, CBLAS_CTYPE *y, int incy, CBLAS_CTYPE *A);

void cblas_zgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *y, int incy);
void cblas_zgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, int m, int n, int kL, int kU, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *y, int incy);
void cblas_ztrmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx);
void cblas_ztbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx);
void cblas_ztpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_ZTYPE *A, CBLAS_ZTYPE *x, int incx);
void cblas_ztrsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx);
void cblas_ztbsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, int k, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx);
void cblas_ztpsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int n, CBLAS_ZTYPE *A, CBLAS_ZTYPE *x, int incx);
void cblas_zhemv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *y, int incy);
void cblas_zhbmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, int k, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *y, int incy);
void cblas_zhpmv(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *y, int incy);
void cblas_zgeru(CBLAS_ORDER order, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *A, int ldA);
void cblas_zgerc(CBLAS_ORDER order, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *A, int ldA);
void cblas_zher(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *A, int ldA);
void cblas_zhpr(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, double alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *A);
void cblas_zher2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *A, int ldA);
void cblas_zhpr2(CBLAS_ORDER order, CBLAS_UPLO uplo, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *x, int incx, CBLAS_ZTYPE *y, int incy, CBLAS_ZTYPE *A);

void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC);
void cblas_ssymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC);
void cblas_ssyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, float *A, int ldA, float beta, float *C, int ldC);
void cblas_ssyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta, float *C, int ldC);
void cblas_strmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, float alpha, float *A, int ldA, float *B, int ldB);
void cblas_strsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, float alpha, float *A, int ldA, float *B, int ldB);

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC);
void cblas_dsymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC);
void cblas_dsyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, double alpha, double *A, int ldA, double beta, double *C, int ldC);
void cblas_dsyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, double alpha, double *A, int ldA, double *B, int ldB, double beta, double *C, int ldC);
void cblas_dtrmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, double alpha, double *A, int ldA, double *B, int ldB);
void cblas_dtrsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, double alpha, double *A, int ldA, double *B, int ldB);

void cblas_cgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB, CBLAS_CTYPE *beta, CBLAS_CTYPE *C, int ldC);
void cblas_csymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB, CBLAS_CTYPE *beta, CBLAS_CTYPE *C, int ldC);
void cblas_csyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *beta, CBLAS_CTYPE *C, int ldC);
void cblas_csyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB, CBLAS_CTYPE *beta, CBLAS_CTYPE *C, int ldC);
void cblas_ctrmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB);
void cblas_ctrsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB);
void cblas_chemm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB, CBLAS_CTYPE *beta, CBLAS_CTYPE *C, int ldC);
void cblas_cherk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, CBLAS_CTYPE *A, int ldA, float beta, CBLAS_CTYPE *C, int ldC);
void cblas_cher2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_CTYPE *alpha, CBLAS_CTYPE *A, int ldA, CBLAS_CTYPE *B, int ldB, float beta, CBLAS_CTYPE *C, int ldC);

void cblas_zgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *C, int ldC);
void cblas_zsymm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *C, int ldC);
void cblas_zsyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *C, int ldC);
void cblas_zsyr2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *C, int ldC);
void cblas_ztrmm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB);
void cblas_ztrsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB);
void cblas_zhemm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB, CBLAS_ZTYPE *beta, CBLAS_ZTYPE *C, int ldC);
void cblas_zherk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, double alpha, CBLAS_ZTYPE *A, int ldA, double beta, CBLAS_ZTYPE *C, int ldC);
void cblas_zher2k(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, CBLAS_ZTYPE *alpha, CBLAS_ZTYPE *A, int ldA, CBLAS_ZTYPE *B, int ldB, double beta, CBLAS_ZTYPE *C, int ldC);
    
#ifdef __cplusplus
}
#endif

#endif
