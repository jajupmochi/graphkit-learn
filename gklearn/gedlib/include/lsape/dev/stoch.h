// -----------------------------------------------------------   
/** \file stoch.h
 *  \brief Dealing with bistochastic matrices (Sinkhorn-Knopp, random generation)
 *  \author Sebastien Bougleux (Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC, Caen, France)
*/
/* ----------------------------------------------------------- 
   This file is part of LSAPE.
   
   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.

   -----------------------------------------------------------
   Creation: October 19 2017
   Last modif: March 2018
   
*/

#ifndef __STOC_H__
#define __STOC_H__

#include <cstring>
#include <limits>
#include <cmath>
#include <random>

// -------------------------------------------------------------------------------
template <typename DT,typename IT>
void sinkhornKnopp_c1(const DT *A, const IT &nbr, const IT &nbc, DT *c)
{
  const DT *pt = A;
  DT *spt = c, un = (DT)1;
  IT i,j;

  for (j = 0; j < nbc; j++, ++spt)
  {
    *spt = 0;
    for (i = 0; i < nbr; i++, ++pt)
    {
      *spt += *pt;
    }
    *spt = un / *spt;
  }
}
// -------------------------------------------------------------------------------
template <typename DT,typename IT>
void sinkhornKnopp_c(const DT *A, const IT &nbr, const IT &nbc, const DT *r, DT *c, DT &errc)
{
  const DT *pt = A, *ptr = NULL, un = (DT)1;
  DT *spt = c, clastj = 0;
  IT i,j;

  errc = (DT)0;

  for (j = 0; j < nbc; j++, ++spt)
  {
    clastj = *spt;
    *spt = 0;
    ptr = r;
    for (i = 0; i < nbr; i++, ++pt, ++ptr)
    {
      *spt += *pt * *ptr;
    }
    errc += std::abs((clastj * *spt) - 1);
    *spt = un / *spt;
  }
}
// -------------------------------------------------------------------------------
template <typename DT,typename IT>
void sinkhornKnopp_r(const DT *A, const IT &nbr, const IT &nbc, const DT *c, DT *r)
{
  const DT *pt = NULL, *ptc = NULL, un = (DT)1;
  DT *rpt = r;
  IT i,j;

  for (i = 0; i < nbr; i++, ++rpt)
  {
    *rpt = 0;
    ptc = c;
    pt = A+i;
    for (j = 0; j < nbc; j++, pt+=nbr, ++ptc)
    {
      *rpt += *pt * *ptc;
    }
    *rpt = un / *rpt;
  }

}
// -------------------------------------------------------------------------------
template <typename DT,typename IT>
void sinkhornKnopp(DT *A, const IT &nbr, const IT &nbc, DT tol = 1e-8, int nbit = 50)
{
  DT *c = new DT[nbc], *r = new DT[nbr], errc = std::numeric_limits<DT>::max();
  
  // init and 1st iteration
  sinkhornKnopp_c1(A,nbr,nbc,c);
  sinkhornKnopp_r(A,nbr,nbc,c,r);
  
  // other iterations
  int nb = 2;
  for (; nb < nbit && errc > tol; nb++)
  {
    sinkhornKnopp_c(A,nbr,nbc,r,c,errc);
    sinkhornKnopp_r(A,nbr,nbc,c,r);
  }
  
  // reconstruction
  DT *pt = A;
  const DT *ptr = r, *ptc = c;
  int i, j;

  for (j = 0; j < nbc; j++, ++ptc)
  {
    for (i = 0; i < nbr; i++, ++pt, ++ptr)
    {
      *pt = (*pt) * (*ptr) * (*ptc);
    }
    ptr = r;
  }
  
  delete[] r; delete[] c;
}
// -----------------------------------------------------------
template <typename DT,typename IT>
void randVecStoch(const IT &n, DT *v)
{
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<double> distribution(0,1.0);
  for (int i = 0; i < n; i++) v[i] = distribution(e2);
}
// -----------------------------------------------------------
template <typename DT,typename IT>
void randStoch(const IT &n, DT *A)
{
  DT *Z = new DT[n*n], sum = 0;
  DT *ptA = A, *ptZ = Z, *ptZZ = NULL, *ptAA = NULL;
  int i = 0;
  
  for (int j = 0; j < n; j++, ptA+=n, ptZ+=n)
  {
    randVecStoch(n,ptA);
    randVecStoch(n,ptZ);
    ptAA = ptA; ptZZ = ptZ;
    sum = 0;
    for (i = 0; i < n; i++, ++ptAA, ++ptZZ)
    {
      *ptAA = (*ptAA) * (*ptAA) + (*ptZZ) * (*ptZZ);
      sum += *ptAA;
    }
    ptAA = ptA;
    for (i = 0; i < n; i++, ++ptAA) *ptAA /= sum;
  }
  
  delete[] Z;
}
// -----------------------------------------------------------
template <typename DT,typename IT>
void randBiStoch(const IT &n, DT *A, DT tol = 1e-8, int nbit = 50)
{
  randStoch(n,A);
  sinkhornKnopp(A,n,n,tol,nbit);
}

#endif
