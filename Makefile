####
#### Top level Makefile for COBLAS library
####

CC=cc
CFLAGS=-O3 -std=c99
PREFIX=/usr/local
LIB=libcoblas.a
INC=coblas.h
INSTALL=install

default: lib/$(LIB)

all: default doc 

lib/$(LIB):
	@cd src;$(MAKE) CC="$(CC)" CFLAGS="$(CFLAGS)" TARGET="$(LIB)"

install: lib/$(LIB)
	@cd src;$(MAKE) CC="$(CC)" CFLAGS="$(CFLAGS)" TARGET="$(LIB)" INSTALL="$(INSTALL)" PREFIX="$(PREFIX)" install

clean:
	@cd src;$(MAKE) clean
