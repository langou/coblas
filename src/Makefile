####
#### Makefile for COBLAS library
####

CC=cc
CFLAGS=-O3 -std=c99
TARGET=libcoblas.a
LIBTOOL=ar cr
INSTALL=install
INC=coblas.h
PREFIX=/usr/local
LIBS=

OBJ=cblas_xerbla.o \
cblas_saxpy.o cblas_isamax.o cblas_sasum.o cblas_scopy.o cblas_sdot.o cblas_snrm2.o \
cblas_srot.o cblas_srotg.o cblas_sscal.o cblas_sswap.o cblas_sdsdot.o cblas_srotm.o \
cblas_srotmg.o \
cblas_daxpy.o cblas_idamax.o cblas_dasum.o cblas_dcopy.o cblas_ddot.o cblas_dnrm2.o \
cblas_drot.o cblas_drotg.o cblas_dscal.o cblas_dswap.o cblas_dsdot.o cblas_drotm.o \
cblas_drotmg.o \
cblas_caxpy.o cblas_icamax.o cblas_scasum.o cblas_ccopy.o cblas_cdotc_sub.o cblas_cdotu_sub.o \
cblas_scnrm2.o cblas_csscal.o cblas_cscal.o cblas_cswap.o \
cblas_zaxpy.o cblas_izamax.o cblas_dzasum.o cblas_zcopy.o cblas_zdotc_sub.o cblas_zdotu_sub.o \
cblas_dznrm2.o cblas_zdscal.o cblas_zscal.o cblas_zswap.o \
cblas_sgemv.o cblas_sgbmv.o cblas_ssymv.o cblas_ssbmv.o cblas_sspmv.o cblas_strmv.o \
cblas_stbmv.o cblas_stpmv.o cblas_strsv.o cblas_stbsv.o cblas_stpsv.o cblas_sger.o \
cblas_ssyr.o cblas_sspr.o cblas_ssyr2.o cblas_sspr2.o \
cblas_dgemv.o cblas_dgbmv.o cblas_dsymv.o cblas_dsbmv.o cblas_dspmv.o cblas_dtrmv.o \
cblas_dtbmv.o cblas_dtpmv.o cblas_dtrsv.o cblas_dtbsv.o cblas_dtpsv.o cblas_dger.o \
cblas_dsyr.o cblas_dspr.o cblas_dsyr2.o cblas_dspr2.o \
cblas_cgemv.o cblas_cgbmv.o cblas_chemv.o cblas_chbmv.o cblas_chpmv.o cblas_ctrmv.o \
cblas_ctbmv.o cblas_ctpmv.o cblas_ctrsv.o cblas_ctbsv.o cblas_ctpsv.o cblas_cgerc.o \
cblas_cgeru.o cblas_cher.o cblas_chpr.o cblas_cher2.o cblas_chpr2.o \
cblas_zgemv.o cblas_zgbmv.o cblas_zhemv.o cblas_zhbmv.o cblas_zhpmv.o cblas_ztrmv.o \
cblas_ztbmv.o cblas_ztpmv.o cblas_ztrsv.o cblas_ztbsv.o cblas_ztpsv.o cblas_zgerc.o \
cblas_zgeru.o cblas_zher.o cblas_zhpr.o cblas_zher2.o cblas_zhpr2.o \
cblas_sgemm.o cblas_ssymm.o cblas_ssyrk.o cblas_ssyr2k.o cblas_strmm.o cblas_strsm.o \
cblas_dgemm.o cblas_dsymm.o cblas_dsyrk.o cblas_dsyr2k.o cblas_dtrmm.o cblas_dtrsm.o \
cblas_cgemm.o cblas_csymm.o cblas_csyrk.o cblas_csyr2k.o cblas_ctrmm.o cblas_ctrsm.o \
cblas_chemm.o cblas_cherk.o cblas_cher2k.o \
cblas_zgemm.o cblas_zsymm.o cblas_zsyrk.o cblas_zsyr2k.o cblas_ztrmm.o cblas_ztrsm.o \
cblas_zhemm.o cblas_zherk.o cblas_zher2k.o 

all: ../lib/$(TARGET)

.c.o:
	$(CC) -c $(CFLAGS) -I../include $<

$(OBJ): ../include/$(INC)

../lib/$(TARGET): $(OBJ)
	$(INSTALL) -d ../lib
	$(LIBTOOL) ../lib/$(TARGET) $(OBJ) $(LIBS)

install: ../lib/$(TARGET)
	$(INSTALL) ../lib/$(TARGET) $(PREFIX)/lib
	$(INSTALL) ../include/$(INC) $(PREFIX)/include

clean:
	rm -f $(OBJ) 
	rm -fr ../lib/$(TARGET)
