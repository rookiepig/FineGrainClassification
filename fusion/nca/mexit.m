%
% compile the nca code
%
% charless fowlkes
% fowlkes@cs.berkeley.edu
% 2005-02-23
%

mex nca.cc CC=g++ COPTIMFLAGS=-O3 CDEBUGFLAGS=-g CFLAGS='-fPIC -ansi -D_GNU_SOURCE -pthread -Wall' CXX=g++ CXXOPTIMFLAGS=-O3 CXXDEBUGFLAGS=-g CXXFLAGS='-fPIC -ansi -D_GNU_SOURCE -pthread -Wall' 

