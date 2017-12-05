/*
Python wrapper
*/

#include "hungarian-lsape.hh"
#include "hungarian-lsap.hh"

#include <cstdio>

extern "C" int lsap(double * C, const int nm, long * rho, long * varrho){
  double * u = new double[nm];
  double * v = new double[nm];

  int * rho_int = new int[nm];
  int * varrho_int = new int[nm];

  hungarianLSAP(C,nm,nm,rho_int,u,v,varrho_int);
  //Find a better way to do
  for (int i =0;i<nm;i++){
    rho[i] = (long)(rho_int[i]);
    varrho[i] = (long)(varrho_int[i]);
  }  
  return 0;
}



extern "C" int * lsape(double * C, const int n, const int m, long * rho, long * varrho){
  double * u = new double[n];
  double * v = new double[m];

  int * rho_int = new int[n];
  int * varrho_int = new int[m];

  hungarianLSAPE(C,n,m,rho_int,varrho_int,u,v);
  for (int i =0;i<n;i++)
    rho[i] = (long)(rho_int[i]);

  for (int i =0;i<m;i++)
    varrho[i] = (long)(varrho_int[i]);
  
  return 0;
}
