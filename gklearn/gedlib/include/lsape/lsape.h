// -----------------------------------------------------------   
/** \file lsape.h
 *  \brief Main file for solving the LSAPE / minimal-cost error-correcting bipartite graph matching problem
 * \author Sebastien Bougleux (Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC, Caen, France)
*/
   
/* -----------------------------------------------------------
   
   This file is part of LSAPE.
   
   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.

   -----------------------------------------------------------
   
   Creation: Oct. 2017
   Last modif: March 2018
   
*/

#ifndef __LSAPE_H_
#define __LSAPE_H_

#include <cstring>
#include <limits>
#include <queue>
#include "lsap.h"
#include "dev/deftypes.h"
#include "dev/hungarian-lsape.h"
#include "dev/enum-ec-matchings.h"
#include "dev/greedy-lsape.h"
#include "dev/stoch.h"

namespace lsape {

  /*// -----------------------------------------------------------
    template <typename DT, typename IT>
    DT ecPermCost(const DT *C, const IT &nr, const IT &nc, const IT *rho, const IT *varrho)
    {
    DT sumcost = 0;
    const IT n = nr-1, m = nc-1;
    for (int i = 0; i < n; i++) sumcost += C[rho[i]*nr+i];
    for (int j = 0; j < m; j++)
    if (varrho[j] == n) sumcost += C[j*nr+varrho[j]];
    return sumcost;
    }
  */
  // -----------------------------------------------------------
  /** @brief Test if triangular inequalities are satisfied
   *  @param[in] C nrxnc edit cost matrix represented as an array
   *  @param[in] nr Number of rows
   *  @param[in] nc Number of columns
   *  @return true if triangular inequalities are satisfied, false else
   */
  template <typename DT>
    bool lsapeTriangleInequality(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc)
    {
      const LSAPE_IndexType m = nc-1, n = nr-1;
      const LSAPE_IndexType n1m = (n+1)*m;
      LSAPE_IndexType i, j;
      const DT *ptC = C, *ptIns;
      
      for (j = 0; j < m; j++, ++ptC)
      {
	ptIns = ptC + n;
	for (i = 0; i < n; i++, ++ptC)
	{
	  if (*ptC > *(ptC+n1m) + *ptIns) return false;
	}
      }
      return true;
    }

  // -----------------------------------------------------------
  /** @brief Test if triangular inequality for a pair (i,j) is satisfied
   *  @param[in] C nrxnc edit cost matrix represented as an array
   *  @param[in] nr Number of rows
   *  @param[in] nc Number of columns
   *  @param[in] i Row to test
   *  @param[in] j Column to test
   *  @return true if triangular inquality is satisfied, false else
   */
  template <typename DT>
    bool lsapeTriangleInequality(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType i, LSAPE_IndexType j)
    { return (C[j*nr+i] <= C[(nc-1)*nr+i] + C[j*nr+(nr-1)]); }

  // -----------------------------------------------------------
  /*
  template <typename DT, typename IT>
  void ecMtxStochasticBarycenter(IT nr, IT nc, DT *J)
  {
  const DT val = ((DT)2) / (nr + nc);
  std::memset(J,val,sizeof(DT)*(nr*nc-1));
  J[nr*nc-1] = 0;
  }
  */
  // -----------------------------------------------------------
  
  template <typename DT>
    DT* randBiStochExt(const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT tol = 1e-8, int nbit = 50)
  {
    const LSAPE_IndexType nm = nr + nc, n1 = nr+1, m1 = nc+1;
    LSAPE_IndexType i, j;
    DT *X = new DT[nm*nm], *A = new DT[m1], *Z = new DT[m1], sum;
    DT *ptA = A, *ptZ = Z, *ptX = X;
  
    for (j = 0; j < nc; j++)
      {
	randVecStoch(n1,A);
	randVecStoch(n1,Z);
	ptA = A; ptZ = Z;
	sum = 0;
	for (i = 0; i < n1; i++, ++ptA, ++ptZ)
	  {
	    *ptA = (*ptA) * (*ptA) + (*ptZ) * (*ptZ);
	    sum += *ptA;
	  }
	ptA = A;
	for (i = 0; i < nr; i++, ++ptA, ++ptX) *ptX = *ptA / sum;
	for (i = nr; i < nm; i++, ++ptX) *ptX = 0;
	X[j*nm+j+nr] = *ptA / sum;
      }

    for (j = nc; j < nm; j++)
      {
	randVecStoch(m1,A);
	randVecStoch(m1,Z);
	ptA = A; ptZ = Z;
	sum = 0;
	for (i = 0; i < m1; i++, ++ptA, ++ptZ)
	  {
	    *ptA = (*ptA) * (*ptA) + (*ptZ) * (*ptZ);
	    sum += *ptA;
	  }
	ptA = A;
	for (i = 0; i < nr; i++, ++ptX) *ptX = 0;
	for (i = nr; i < nm; i++, ++ptA, ++ptX) *ptX = *ptA / sum;
	X[j*nm+j-nc] = *ptA / sum;
      }

    sinkhornKnopp(X,nm,nm,tol,nbit);

    delete[] A, delete[] Z;
    return X;
  }

  // -----------------------------------------------------------
  /** \brief Extend a (n+1)x(m+1) LSAPE instance to an equivalent (n+m)x(m+n) LSAP instance
   *  \param[in] C nrowsxncols edit cost matrix
   *  \param[in] nrows number of rows (last row corresponds to the null element)
   *  \param[in] ncols number of colums (last column corresponds to the null element)
   *  \param[out] Cext (nrows-1+ncols-1)x(ncols-1+nrows-1) extended cost matrix (must be previously allocated)
   *  \param[in] valmx value given to outer diagonal elements of Cext (forbidden assignments)
   */
  template <typename DT>
    void lsapeInstanceEBP(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, DT *Cext, DT valmx = std::numeric_limits<DT>::max())
  {
    const LSAPE_IndexType msub = ncols-1, nsub = nrows-1;
    LSAPE_IndexType i, j;
    const LSAPE_IndexType mpn = msub+nsub;
    // copy subsitutions
    for (j = 0; j < msub; j++) std::memcpy(Cext+j*mpn,C+j*nrows,sizeof(DT)*nsub);
    // copy insertions
    for (j = 0; j < msub; j++)
      for (i = nsub; i < mpn; i++)
	if (i != j+nsub) Cext[j*mpn+i] = valmx;
	else Cext[j*mpn+i] = C[j*nrows+nsub];
    // copy removals
    for (j = msub; j < mpn; j++)
      for (i = 0; i < nsub; i++)
	if (i+msub != j) Cext[j*mpn+i] = valmx;
	else Cext[j*mpn+i] = C[msub*nrows+i];
    // set completness
    for (i = nsub; i < mpn; i++)
      for (j = msub; j < mpn; j++)
	Cext[j*mpn+i] = 0;
  }

  // -----------------------------------------------------------
  template <typename DT>
    void lsapeEBP(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType npm = nr+nc-2;
    LSAPE_IndexType i = 0, j;
    const LSAPE_IndexType npm2 = npm*npm;
    DT *Cext = new DT[npm2];

    // find max of C
    DT mx = std::numeric_limits<DT>::min(), mxx = 10;
    for (const DT *ptC = C; i < nc*nr; i++, ++ptC) if (*ptC > mx) mx = *ptC;

    // construct the instance of the symmetric LSAP
    mxx += mx;
    lsapeInstanceEBP(C,nr,nc,Cext,mxx);
  
    DT *ux = new DT[npm], *vx = new DT[npm];
    LSAPE_IndexType *rhox = new LSAPE_IndexType[npm], *varrhox = NULL;
    if (varrho) varrhox = new LSAPE_IndexType[npm];

    hungarianLSAP(Cext,npm,npm,rhox,ux,vx,varrhox,init_type);
  
    // reconstruction
    const LSAPE_IndexType n = nr-1, m = nc-1;
    for (i = 0; i < n; i++)
      {
	rho[i] = (rhox[i] >= m ? m : rhox[i]);
	u[i] = ux[i] + vx[m+i];
      }
    if (varrho)
      for (j = 0; j < m; j++)
	{
	  varrho[j] = (varrhox[j] >= n ? n : varrhox[j]);
	  v[j] = vx[j] + ux[n+j];
	}
    else for (j = 0; j < m; j++) v[j] = vx[j] + ux[n+j];

    delete[] Cext; delete[] rhox; if (varrho) delete[] varrhox;
    delete[] ux; delete[] vx;
  }

  // -----------------------------------------------------------
  /** \brief Reduce a (n+1)x(m+1) LSAPE instance to an equivalent nxm LSAP instance for RBP
   *  \param[in] C nxm edit cost matrix
   *  \param[in] nr number of rows (last row correspond to the null element)
   *  \param[in] nc number of colums (last column correspond to the null element)
   *  \param[out] Cred (n-1)x(m-1) reduced cost matrix (must be previously allocated)
   *  \return minimum value of reduced cost matrix Cred
   */
  template <typename DT>
    DT lsapeInstanceFLWC(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT *Cred)
  {
    const LSAPE_IndexType m = nc-1, n = nr-1;
    LSAPE_IndexType i;
    const LSAPE_IndexType mnr = m*nr;
    const DT *ptC = C, *ptIns = C, *ptEnd = C + nr*nc - nr, *ptRem = C;
    DT *ptR = Cred, cd, minCred = std::numeric_limits<DT>::max();

    if (n < m)
      {
	for (; ptC != ptEnd; ++ptC)
	  for (ptIns = ptC + n, i = 0; i < n; i++, ++ptC, ++ptR)
	    {
	      cd = *ptC - *ptIns;
	      ptRem = C+mnr+i;
	      *ptR = (cd < *ptRem ? cd : *ptRem);
	      if (*ptR < minCred) minCred = *ptR;
	    }
      }
    else if (n == m)
      {
	for (; ptC != ptEnd; ++ptC)
	  for (ptIns = ptC + n, i = 0; i < n; i++, ++ptC, ++ptR)
	    {
	      cd = C[mnr+i] + *ptIns;
	      *ptR = (cd < *ptC ? cd : *ptC);
	      if (*ptR < minCred) minCred = *ptR;
	    }
      }
    else
      {
	for (; ptC != ptEnd; ++ptC)
	  for (i = 0, ptIns = ptC + n; i < n; i++, ++ptC, ++ptR)
	    {
	      cd = *ptC - C[mnr+i];
	      *ptR = (cd < *ptIns ? cd : *ptIns);
	      if (*ptR < minCred) minCred = *ptR;
	    }
      }
    return minCred;
  }

  // -----------------------------------------------------------
  template <typename DT>
    void lsapeFLWC(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType n = nr-1, m = nc-1;
    LSAPE_IndexType i, j;
    DT *Cred = new DT[n*m], *vec = NULL;
  
    bool delvarrho = false;
    if (varrho == NULL) { delvarrho = true; varrho = new LSAPE_IndexType[m]; }

    DT mn = lsapeInstanceFLWC(C,nr,nc,Cred);

    // Cred may contain negative values -> positive
    const DT *citend = Cred + n*m;
    if (n != m && mn < 0) for (DT *cit = Cred; cit != citend; ++cit) *cit -= mn;

    hungarianLSAP(Cred,n,m,rho,u,v,varrho,init_type);
  
    // reconstruct primal-dual solutions
    if (n > m)
      {
	if (mn < 0)
	  {
	    for (j = 0, vec = v; j < m; j++, ++vec)
	      {
		i = varrho[j];
		if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
		*vec += mn;
	      }
	  }
	else
	  {
	    for (j = 0, vec = v; j < m; j++, ++vec)
	      {
		i = varrho[j];
		if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
	      }
	  }
	for (i = 0, vec = u; i < n; i++, ++vec) { /*if (rho[i] == m) rho[i] = m;*/ *vec += C[m*nr+i]; }
      }
    else if (n < m) // --------
      {
	if (mn < 0)
	  {
	    for (i = 0, vec = u; i < n; i++, ++vec)
	      {
		j = rho[i];
		if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
		*vec += mn;
	      }
	  }
	else
	  {
	    for (i = 0, vec = u; i < n; i++, ++vec)
	      {
		j = rho[i];
		if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
	      }
	  }
	for (j = 0, vec = v; j < m; j++, ++vec) { /*if (varrho[j] == n) varrho[j] = m;*/ *vec += C[j*nr+n]; }
      }
    else  // n == m
      {
	if (mn < 0)
	  for (i = 0, vec = u; i < n; i++, ++vec)
	    {
	      j = rho[i];
	      if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
	      *vec += mn;
	    }
	else
	  for (i = 0, vec = u; i < n; i++, ++vec)
	    {
	      j = rho[i];
	      if (!lsapeTriangleInequality(C,nr,nc,i,j)) { rho[i] = m; varrho[j] = n; }
	    }
      }
  
    delete[] Cred;
    if (delvarrho) { delete[] varrho; varrho = NULL; }
  }

  // -----------------------------------------------------------
  /** \brief Reduce a (n+1)x(m+1) LSAPE instance to an equivalent nxm LSAP instance for RBP
   *  \param[in] C nxm edit cost matrix
   *  \param[in] nr number of rows (last row correspond to the null element)
   *  \param[in] nc number of colums (last column correspond to the null element)
   *  \param[out] Cred (n-1)x(m-1) reduced cost matrix (must be previously allocated)
   */
  template <typename DT>
    DT lsapeInstanceFLCC(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT *Cred)
  {
    const LSAPE_IndexType m = nc-1, n = nr-1;
    const LSAPE_IndexType mnr = m*nr;
    LSAPE_IndexType i, j;
    const DT *ptC = C, *ptIns = C, *ptEnd = C + nr*nc - nr, *ptRem = C;
    DT *ptR = Cred, cd, minCred = std::numeric_limits<DT>::max();

    if (n < m) // no removal
      for (; ptC != ptEnd; ++ptC)
	for (ptIns = ptC + n, i = 0; i < n; i++, ++ptC, ++ptR)
	{
	  *ptR = *ptC - *ptIns;
	  if (*ptR < minCred) minCred = *ptR;
	}
    else if (n > m) // no insertion
      for (; ptC != ptEnd; ++ptC)
	for (i = 0, ptIns = ptC + n; i < n; i++, ++ptC, ++ptR)
	{
	  *ptR = *ptC - C[mnr+i];
	  if (*ptR < minCred) minCred = *ptR;
	}
    else // only substitutions
      for (; ptC != ptEnd; ++ptC)
	for (ptIns = ptC + n, i = 0; i < n; i++, ++ptC, ++ptR)
	{
	  *ptR = *ptC;
	  if (*ptR < minCred) minCred = *ptR;
	}
    return minCred;
  }
  
  // -----------------------------------------------------------
  template <typename DT>
    void lsapeFLCC(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType n = nr-1, m = nc-1;
    LSAPE_IndexType i, j;
    DT *Cred = new DT[n*m], *vec = NULL;
  
    bool delvarrho = false;
    if (varrho == NULL) { delvarrho = true; varrho = new LSAPE_IndexType[m]; }

    DT mn = lsapeInstanceFLWC(C,nr,nc,Cred);

    // Cred may contain negative values -> positive
    const DT *citend = Cred + n*m;
    if (n != m && mn < 0) for (DT *cit = Cred; cit != citend; ++cit) *cit -= mn;

    hungarianLSAP(Cred,n,m,rho,u,v,varrho,init_type);
  
    // reconstruct primal-dual solutions
    if (n > m)
    {
      if (mn < 0) for (j = 0, vec = v; j < m; j++, ++vec) *vec += mn;
      for (i = 0, vec = u; i < n; i++, ++vec) *vec += C[m*nr+i];
    }
    else if (n < m) // --------
    {
      if (mn < 0) for (i = 0, vec = u; i < n; i++, ++vec) *vec += mn;
      for (j = 0, vec = v; j < m; j++, ++vec) *vec += C[j*nr+n];
    }
    else  // n == m
      if (mn < 0) for (i = 0, vec = u; i < n; i++, ++vec) *vec += mn;
  
    delete[] Cred;
    if (delvarrho) { delete[] varrho; varrho = NULL; }
  }

  // -----------------------------------------------------------
  /** \brief Reduce a (n+1)x(m+1) LSAPE instance to an equivalent nxm LSAP instance for FBP
   *  \param[in] C nxm edit cost matrix
   *  \param[in] nr number of rows (last row correspond to the null element)
   *  \param[in] nc number of colums (last column correspond to the null element)
   *  \param[out] Cred (n-1)x(m-1) reduced cost matrix (must be previously allocated)
   */
  template <typename DT>
    void lsapeInstanceFBP(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT *Cred)
  {
    LSAPE_IndexType m = nc-1, n = nr-1, i, j;
    const DT *ptC = C, *ptIns;
    DT *ptR = Cred;

    for (j = 0; j < m; j++, ++ptC)
    {
      ptIns = ptC + n;
      for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC - C[m*nr+i] - *ptIns;
    }
  }

  // -----------------------------------------------------------
  template <typename DT>
    void lsapeFBP(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType n = nr-1, m = nc-1;
    LSAPE_IndexType i, j;
    DT *Cred = new DT[n*m];

    lsapeInstanceFBP<DT>(C,nr,nc,Cred);

    // Cred contains negative values -> positive
    DT mn = std::numeric_limits<DT>::max(), mnval = 0;
    for (j = 0; j < m; j++)
      for (i = 0; i < n; i++)
      {
	const DT &cr = Cred[j*n+i];
	if (cr < mn) mn = cr;
      }
    if (mn < 0)
    {
      for (j = 0; j < m; j++)
	for (i = 0; i < n; i++)
	  Cred[j*n+i] -= mn;
      mnval = mn;
    }

    hungarianLSAP(Cred,n,m,rho,u,v,varrho,init_type);
  
    // reconstruction
    if (n < m) // no removal
    {
      if (varrho) for (j = 0; j < m; j++) { /*if (varrho[j] == n) varrho[j] = n*/;  v[j] += C[j*nr+n]; }
      else for (j = 0; j < m; j++) v[j] += C[j*nr+n];
      for (i = 0; i < n; i++) u[i] += C[m*nr+i] + mnval;
    }
    else if (n > m) // no insertion
    {
      for (i = 0; i < n; i++) { /*if (rho[i] == m) rho[i] = m;*/ u[i] += C[m*nr+i]; }
      for (j = 0; j < m; j++) v[j] += C[j*nr+n] + mnval;
    }
    else // substitutions
    {
      for (i = 0; i < n; i++) u[i] += C[m*nr+i] + mnval;
      for (j = 0; j < m; j++) v[j] += C[j*nr+n];
    }
  
    delete[] Cred;
  }

  // -----------------------------------------------------------
  /** \brief Extend a (n+1)x(m+1) LSAPE instance to an equivalent max{n,m}xmax{n,m} LSAP instance for FBP0
   *  \param[in] C (n+1)x(m+1) edit cost matrix
   *  \param[in] nr number of rows nr=n+1 (last row correspond to the null element)
   *  \param[in] nc number of colums nc=m+1 (last column correspond to the null element)
   *  \param[out] Cext max{n,m}xmax{n,m} reduced cost matrix (must be previously allocated)
   */
  template <typename DT>
    void lsapeInstanceFBP0(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT *Cext)
  {
    const LSAPE_IndexType m = nc-1, n = nr-1, mn = (m >= n) ? (m - n) : (n - m);
    LSAPE_IndexType i, j;
    const DT *ptC = C;
    DT *ptR = Cext, *ptRem = NULL, zero = 0;

    if (n < m)
    {
      // substitutions
      for (j = 0; j < m; j++, ++ptC, ptR+=mn)
      {
	const DT &ins = C[j*nr+n];
	for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC - ins - C[m*nr+i];
      }
      // insertions
      ptR = Cext+n;
      for (j = 0; j < m; j++, ptR+=n)
	for (i = 0; i < mn; i++, ptR++)
	  *ptR = zero;
    }
    else 
      if (n > m)
      {
	// substitutions
	for (j = 0; j < m; j++, ++ptC)
	{
	  const DT &ins = C[j*nr+n];
	  for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC - ins - C[m*nr+i];
	}
	// removals
	ptRem = Cext+n*m;
	for (j = 0; j < mn; j++, ptRem+=n) std::memset(ptRem,0,sizeof(DT)*n);
      }
      else // only substitutions
      {
	for (j = 0; j < m; j++, ++ptC)
	{
	  const DT &ins = C[j*nr+n];
	  for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC - ins - C[m*nr+i];
	}
      }
  }

  // -----------------------------------------------------------
  template <typename DT>
    void lsapeFBP0(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType n = nr-1, m = nc-1;
    LSAPE_IndexType i, j;
    LSAPE_IndexType mxnm = std::max(n,m);
    DT *Cext = new DT[mxnm*mxnm];

    lsapeInstanceFBP0<DT>(C,nr,nc,Cext);
  
    // the instance is negative -> find min and translate
    DT mn = std::numeric_limits<DT>::max(), mnval = 0;
    for (j = 0; j < m; j++)
      for (i = 0; i < n; i++)
       {
	 const DT &c = Cext[j*mxnm+i];
	 if (c < mn) mn = c;
       }
    if (mn < 0)
    {
      DT *pt = Cext;
      for (i = 0; i < mxnm*mxnm; i++, ++pt) *pt -= mn;
      mnval = mn;
    }
  
    if (n <= m) // no removal
    {
      LSAPE_IndexType *rhox = new LSAPE_IndexType[mxnm];
      DT *ux = new DT[mxnm];
      
      hungarianLSAP(Cext,mxnm,mxnm,rhox,ux,v,varrho,init_type);
      
      DT lval = (n < m ? ux[n] : 0);
      for (i = 0; i < n; i++) { rho[i] = rhox[i]; u[i] = ux[i] + C[m*nr+i] - lval; }
      if (varrho) for (j = 0; j < m; j++) { if (varrho[j]>n) varrho[j]=n; v[j] += C[j*nr+n] + mnval + lval; }
      else for (j = 0; j < m; j++) v[j] += C[j*nr+n] + mnval + lval;
      
      delete[] rhox; delete[] ux;
    }
    else  // no insertion
    {
      LSAPE_IndexType *varrhox = NULL;
      if (varrho) varrhox = new LSAPE_IndexType[mxnm];
      DT *vx = new DT[mxnm];
      
      hungarianLSAP(Cext,mxnm,mxnm,rho,u,vx,varrhox,init_type);
      
      DT lval = vx[m];
      for (i = 0; i < n; i++) { if (rho[i]>m) rho[i] = m; u[i] += C[m*nr+i] + mnval + lval; }
      if (varrho) for (j = 0; j < m; j++) { varrho[j] = varrhox[j]; v[j] = vx[j] + C[j*nr+n] - lval; }
      else for (j = 0; j < m; j++) v[j] = vx[j] + C[j*nr+n] - lval;

      if (varrho)
	delete[] varrhox;
      delete[] vx;
    }
  
    delete[] Cext;
  }

  // -----------------------------------------------------------
  /** \brief Extend a (n+1)x(m+1) LSAPE instance to an equivalent max{n,m}xmax{n,m} LSAP instance for SFBP
   *  \param[in] C (n+1)x(m+1) edit cost matrix
   *  \param[in] nr number of rows nr=n+1 (last row correspond to the null element)
   *  \param[in] nc number of colums nc+1 (last column correspond to the null element)
   *  \param[out] Cext max{n,m}xmax{n,m} reduced cost matrix (must be previously allocated)
   */
  template <typename DT>
    void lsapeInstanceSFBP(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, DT *Cext)
  {
    const LSAPE_IndexType m = nc-1, n = nr-1, mn = (m >= n) ? (m - n) : (n - m);
    LSAPE_IndexType i, j;
    const DT *ptC = C, *ptIns = NULL;
    DT *ptR = Cext, *ptRem = NULL;

    if (n < m)
    {
      // substitutions
      for (j = 0; j < m; j++, ++ptC, ptR+=mn)
	for (i = 0; i < n; i++, ++ptC, ++ptR)
	  *ptR = *ptC;
      // insertions
      ptIns = C+n;
      ptR = Cext+n;
      for (j = 0; j < m; j++, ptIns+=nr, ptR+=n)
	for (i = 0; i < mn; i++, ptR++)
	  *ptR = *ptIns;
    }
    else 
      if (n > m)
      {
	// substitutions
	for (j = 0; j < m; j++, ++ptC)
	  for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC;
	// removals
	ptRem = Cext+n*m;
	ptC = C+(n+1)*m;
	for (j = 0; j < mn; j++, ptRem+=n) std::memcpy(ptRem,ptC,sizeof(DT)*n);
      }
      else // only substitutions
      {
	for (j = 0; j < m; j++, ++ptC)
	  for (i = 0; i < n; i++, ++ptC, ++ptR) *ptR = *ptC;
      }  
  }

  // -----------------------------------------------------------
  /** \brief SFBP: Solve LSAPE by solving extended LSAP instance of size max{n,m} x max{n,m} (for costs satisfying triangular inequalities)
   *  \param[in] C (n+1)x(m+1) edit cost matrix
   *  \param[in] nr number of rows (last row correspond to the null element)
   *  \param[in] nc number of colums (last column correspond to the null element)
   *  \param[out] rho assignment from rows to columns of C
   *  \param[out] u dual solution associated to rows
   *  \param[out] v dual solution associated to columns
   *  \param[out] varrho assignment from columns to rows
   *  \param[in] init_type initialization type for Hungarian algorithm
   */
  template <typename DT>
    void lsapeSFBP(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, DT *u, DT *v, LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1)
  {
    const LSAPE_IndexType n = nr-1, m = nc-1;
    const LSAPE_IndexType mxnm = std::max(n,m);
    DT *Cext = new DT[mxnm*mxnm];

    lsapeInstanceSFBP<DT>(C,nr,nc,Cext);
  
    if (n <=m) // no removal
    {
      LSAPE_IndexType *rhox = new LSAPE_IndexType[mxnm];
      DT *ux = new DT[mxnm];
      
      hungarianLSAP(Cext,mxnm,mxnm,rhox,ux,v,varrho,init_type);
      
      DT mnval = (n < m ? ux[n] : 0);
      for (int i = 0; i < (int)(n); i++) { rho[i] = rhox[i]; u[i] = ux[i] - mnval; }
      if (varrho) for (int j = 0; j < (int)(m); j++) { if (varrho[j]>n) varrho[j]=n; v[j] += mnval; }
      else for (int j = 0; j < (int)(m); j++) v[j] += mnval;
      
      delete[] rhox; delete[] ux;
    }
    else  // no insertion
    {
      LSAPE_IndexType *varrhox = NULL;
      if (varrho) varrhox = new LSAPE_IndexType[mxnm];
      DT *vx = new DT[mxnm];
      
      hungarianLSAP(Cext,mxnm,mxnm,rho,u,vx,varrhox,init_type);
      
      DT mnval = vx[m];
      for (int i = 0; i < (int)(n); i++) { if (rho[i]>m) rho[i] = m; u[i] += mnval; }
      if (varrho) for (int j = 0; j < (int)(m); j++) { varrho[j] = varrhox[j]; v[j] = vx[j] - mnval; }
      else for (int j = 0; j < (int)(m); j++) v[j] = vx[j] - mnval;
      
      if (varrho)
	delete[] varrhox;
      delete[] vx;
    }
  
    delete[] Cext;
  }

  // -----------------------------------------------------------
  // case with a nrxnc cost matrix with the size of the extended or the reduced model
  template <typename DT>
    void lsapeSolverModel(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, LSAPE_IndexType *varrho, DT *u, DT *v, enum LSAPE_MODEL lsape_model = ECBP, unsigned short init_type = 1)
  {
    switch (lsape_model)
    {
      case ECBP: hungarianLSAPE(C,nr,nc,rho,varrho,u,v,init_type); break;
      case EBP: case SFBP: hungarianLSAP(C,nr,nc,rho,u,v,varrho,init_type); break;
      case FLWC: case FLCC: case FBP: case FBP0:
	{
	  // translate if negative values
	  DT *Ct = new DT[nr*nc];
	  DT minC = C[0], *ptCt = Ct;
	  const DT *pt = C+1, *pte = C+nr*nc;
	  for (; pt != pte; ++pt) if (minC < *pt) minC = *pt;
	  if (minC < 0) for (pt = C; pt != pte; ++ptCt, ++pt) *ptCt = *pt - minC;
	  // solve
	  hungarianLSAP(Ct,nr,nc,rho,u,v,varrho,init_type);
	  // translate back if necessary
	  if (minC < 0)
	  {
	    if (nr < nc) for (LSAPE_IndexType i = 0; i < nr; i++) u[i] += minC;
	    else for (LSAPE_IndexType j = 0; j < nc; j++) v[j] += minC;
	  }
	}
	break;
      default: throw std::runtime_error("unknown value for LSAPE_MODEL in lsapeSolverModel(...)");
    }
  }

  // -----------------------------------------------------------
  // case with cost matrix of size (n+1)x(m+1)=nrxnc
  template <typename DT>
    void lsapeSolver(const DT *C, const LSAPE_IndexType &nr, const LSAPE_IndexType &nc, LSAPE_IndexType *rho, LSAPE_IndexType *varrho, DT *u, DT *v, enum LSAPE_MODEL lsape_model = ECBP, unsigned short init_type = 1)
  {
    switch (lsape_model)
    {
      case ECBP: hungarianLSAPE(C,nr,nc,rho,varrho,u,v,init_type); break;
      case FLWC: lsapeFLWC<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      case EBP: lsapeEBP<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      case FLCC: lsapeFLCC<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      case FBP: lsapeFBP<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      case FBP0: lsapeFBP0<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      case SFBP: lsapeSFBP<DT>(C,nr,nc,rho,u,v,varrho,init_type); break;
      default: throw std::runtime_error("LSAPE_MODEL unknown in lsapeSolver(...)");
    }
  }

  // --------------------------------------------------------------------------------
  // Main function: greedy algorithms for both square and rectangular cost matrices
  /**
   * \brief Compute an approximate solution to the LSAPE as an assignment with low costs, with greedy algorithms given a square or a rectangular cost matrix
   * \param[in] C nxm cost matrix, represented as an array of size \c nm obtained by concatenating its columns (last row and column correspond to null elements)
   * \param[in] n Number of rows of \p C (size of the 1st set)
   * \param[in] m Number of columns of \p C (size of the 2nd set)
   * \param[out] rho Array of size n-1 (must be previously allocated) for the assignment of the rows to the columns, rho[i]=n-1 indicates that i is assigned to the null element, else rho[i]=j indicates that i is assigned to j
   * \param[out] varrho Array of size m-1 (must be previously allocated) for the assignement of the columns to the rows
   * \param[in] greedy_type 0:Basic, 1:Refined, 2: Loss (default), 3: Basic sort, 4: Counting sort (integers only)
   * \return Cost of the assignment
   *
   * \note Adapted from the reference\n
   * Approximate Graph Edit Distance in Quadratic Time. K. Riesen, M. Ferrer, H. Bunke. ACM Transactions on Computational Biology and Bioinformatics, 2015.
   */
  // --------------------------------------------------------------------------------
  template <class DT>
    DT lsapeGreedy(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, LSAPE_IndexType *rho, LSAPE_IndexType *varrho, enum GREEDY_METHOD greedy_method = BASIC)
  {
    switch(greedy_method)
    {
      case BASIC: return greedyBasicLSAPE<DT>(C,nrows,ncols,rho,varrho);
	//case 1: return greedyRefinedLSAPE<DT>(C,nrows,ncols,rho,varrho);
	//case 3: return greedyBasicSortLSAPE<DT>(C,nrows,ncols,rho,varrho);
      case INT_BASIC_SORT: return greedyBasicCountingSortLSAPE<DT>(C,nrows,ncols,rho,varrho);
      default: throw std::runtime_error("GREEDY_METHOD unknown in lsapeGreedy(...)");
    }
  }
  
  // --------------------------------------------------------------------------------
  /**
   * @brief Enumerate solutions to an (n+1)x(m+1) LSAPE instance from an initial solution and a pair of dual solutions
   * @param[in] C nrowsxncols cost matrix
   * @param[in] nrows Number of rows of \p C
   * @param[in] ncols Number of columns of \p C
   * @param[in] nri Number of rows of original instance
   * @param[in] nci Number of columns of original instance
   * @param[in] ksol Number of expected solutions
   * @param[out] r2c A solution to the instance from rows to columns (of size nrows-1)
   * @param[out] c2r The same solution but from columns to rows (of size ncols-1)
   * @param[out] u A dual solution associated to rows (of size nrows)
   * @param[out] v A dual solution associated to columns (of size ncols)
   * @param[out] solutions Array of solutions
   */
  // --------------------------------------------------------------------------------
  template <typename DT>
    void lsapeSolutionsFromOneModel(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const LSAPE_IndexType &nri, const LSAPE_IndexType &nci, const int &ksol,
				    const LSAPE_IndexType *r2c, const LSAPE_IndexType *c2r, const DT *u, const DT *v, std::list<LSAPE_IndexType*> &solutions, LSAPE_MODEL lsape_model = ECBP)
  {
    if (lsape_model == EBP) // EBP, model in (n+m)x(m+n) with n=rows-1, m=ncols-1
    {
      const int npm = nri + nci - 2;
      if (npm != nrows || nrows != ncols) throw std::runtime_error("lsapeSolutionModel(...): we must have nrows=ncols=nri+nci with LSAPE_MODEL == EBP (EBP=2)");
      // construct equality directed graph
      cDigraph<LSAPE_IndexType> edg = equalityDigraph<DT,LSAPE_IndexType>(C,npm,npm,r2c,u,v);      
      // enuerate k solutions, at most
      AllPerfectMatchingsEC<LSAPE_IndexType> apm(edg,nri,nci,solutions);
      apm.enumPerfectMatchings(edg,ksol);
    }
    else throw std::runtime_error("lsapeSolutionModel(...) not yet available with LSAPE_MODEL != EBP (EBP=2)");
  }

  // --------------------------------------------------------------------------------
  /**
   * @brief Enumerate solutions to an (n+1)x(m+1) LSAPE instance from an initial solution and a pair of dual solutions
   * @param[in] C nrowsxncols cost matrix
   * @param[in] nrows Number of rows of \p C
   * @param[in] ncols Number of columns of \p C
   * @param[in] ksol Number of expected solutions
   * @param[out] r2c A solution to the instance from rows to columns (of size nrows-1)
   * @param[out] c2r The same solution but from columns to rows (of size ncols-1)
   * @param[out] u A dual solution associated to rows (of size nrows)
   * @param[out] v A dual solution associated to columns (of size ncols)
   * @param[out] solutions Array of solutions
   */
  // --------------------------------------------------------------------------------
  template <typename DT>
    void lsapeSolutionsFromOne(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const int &ksol,
			       const LSAPE_IndexType *r2c, const LSAPE_IndexType *c2r, const DT *u, const DT *v, std::list<LSAPE_IndexType*> &solutions, LSAPE_MODEL lsape_model = EBP)
  {
    if (lsape_model != EBP) throw std::runtime_error("lsapeSolutionsFromOne(...) not yet available with LSAPE_MODEL != EBP (EBP=2)");
    // transform cost and primal-dual solutions
    const LSAPE_IndexType n = nrows-1, m = ncols-1;
    const LSAPE_IndexType nextd = n+m;
    DT *Cext = new DT[nextd*nextd], *uX = new DT[nextd], *vX = new DT[nextd];
    LSAPE_IndexType *R2C = new LSAPE_IndexType[nextd];
    std::queue<LSAPE_IndexType> liRem;
    
    lsapeInstanceEBP<DT>(C,nrows,ncols,Cext);
    for (int i = 0; i < (int)(n); i++)
    {
      if (r2c[i] == m) { R2C[i] = m+i; }
      else { R2C[i] = r2c[i]; liRem.push(m+i); }
      uX[i] = u[i]; vX[m+i] = 0;
    }
    for (int j = 0; j < (int)(m); j++)
    {
      if (c2r[j] == n) R2C[n+j] = j;
      else { R2C[n+j] = liRem.front(); liRem.pop(); }
      vX[j] = v[j]; uX[n+j] = 0;
    }
		      
    // construct equality directed graph
    cDigraph<LSAPE_IndexType> edg = equalityDigraph<DT,LSAPE_IndexType>(Cext,nextd,nextd,R2C,uX,vX);
    
    // enumerate k solutions, at most
    AllPerfectMatchingsEC<LSAPE_IndexType> apm(edg,n,m,solutions);
    apm.enumPerfectMatchings(edg,ksol);

    // reduce to (n+1)x(m+1) case
    LSAPE_IndexType *match = NULL, i;
    for (typename std::list<LSAPE_IndexType*>::iterator it = solutions.begin(); it != solutions.end(); it++)
    {
     match = *it;
     *it = new LSAPE_IndexType[n];
     for (i = 0; i < n; i++) if (match[i] > m) {(*it)[i] = m;} else {(*it)[i] = match[i];}
     delete match;
     match = NULL;
      //match = (LSAPE_IndexType*)std::realloc(match,n*sizeof(LSAPE_IndexType));
    }
    
    delete[] Cext; delete[] R2C; delete[] uX; delete[] vX;
  }

  // -----------------------------------------------------------
  /**
   * \brief Enumerate solutions to an extended or reduced LSAPE instance
   * @param[in] C nrowsxncols extended or reduced cost matrix
   * @param[in] nrows Number of rows of \p C
   * @param[in] ncols Number of columns of \p C
   * @param[in] nri Number of rows of original instance
   * @param[in] nci Number of columns of original instance
   * @param[out] ksol Maximum number of expected solutions
   * @param[out] solutions Array of solutions
   * @param[in] lsape_model LSAPE model used to find a solution, ECBP by default
   * @return Minimal cost associated to the solutions
   */
  template <typename DT>
    DT lsapeSolutionsModel(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const LSAPE_IndexType &nri, const LSAPE_IndexType &nci,
			   const int &ksol, std::list<LSAPE_IndexType*> &solutions, LSAPE_MODEL lsape_model = ECBP)
  {
    DT mCost = -1;
    if (lsape_model == EBP) // EBP, model in (n+m)x(m+n) with n=rows-1, m=ncols-1
    {
      const LSAPE_IndexType npm = nri + nci - 2;
      if (npm != nrows || nrows != ncols) throw std::runtime_error("lsapeSolutionModel(...): we must have nrows=ncols=nri+nci with LSAPE_MODEL == EBP (EBP=2)");
      LSAPE_IndexType *r2c = new LSAPE_IndexType[npm], *c2r = new LSAPE_IndexType[npm];
      DT *u = new DT[npm], *v = new DT[npm];
      
      // find a solution
      hungarianLSAP(C,npm,npm,r2c,u,v);
      
      // min cost
      for (LSAPE_IndexType i = 0; i < npm; i++) mCost += u[i];
      for (LSAPE_IndexType j = 0; j < npm; j++) mCost += v[j];
      
      // enumerate solutions
      lsapeSolutionsFromOneModel<DT>(C,nrows,ncols,nri,nci,ksol,r2c,c2r,u,v,solutions,EBP);

      delete[] r2c; delete[] c2r; delete[] u; delete[] v;
    }
    else throw std::runtime_error("lsapeSolutionModel(...) not yet available with LSAPE_MODEL != EBP (EBP=2)");
    return mCost;
  }

  // -----------------------------------------------------------
  /**
   * @brief Enumerate solutions to an (n+1)x(m+1) LSAPE instance
   * @param[in] C nrowsxncols cost matrix as an array
   * @param[in] nrows Number of rows of \p C
   * @param[in] ncols Number of columns of \p C
   * @param[in] ksol Number of expected solutions
   * @param[out] solutions List of solutions
   * @param[in] lsape_model LSAPE model used to find a solution, ECBP by default
   * @return Minimal cost associated to the solutions
   */
  template <typename DT>
    DT lsapeSolutions(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const int &ksol, std::list<LSAPE_IndexType*> &solutions, LSAPE_MODEL lsape_model = ECBP)
  {
    const LSAPE_IndexType n = nrows-1, m = ncols-1;
    LSAPE_IndexType *r2c = new LSAPE_IndexType[n], *c2r = new LSAPE_IndexType[m];
    DT *u = new DT[nrows], *v = new DT[ncols], mCost = 0;
      
    // find a solution
    lsapeSolver<DT>(C,nrows,ncols,r2c,c2r,u,v,lsape_model);
    
    // min cost
    for (LSAPE_IndexType i = 0; i < nrows-1; i++) mCost += u[i];
    for (LSAPE_IndexType j = 0; j < ncols-1; j++) mCost += v[j];
    
    // enumerate solutions
    lsapeSolutionsFromOne<DT>(C,nrows,ncols,ksol,r2c,c2r,u,v,solutions,EBP);
      
    delete[] r2c; delete[] u; delete[] v; delete[] c2r;
    return mCost;
  }
  
} // end namespace
 
// -----------------------------------------------------------
#endif
