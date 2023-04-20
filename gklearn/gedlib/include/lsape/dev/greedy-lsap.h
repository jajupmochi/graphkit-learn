// =========================================================================
/** \file greedy-lsap.h
 *  \brief Greedy algorithms for approximatively solving balanced and unbalanced LSAP instances
 * \author Sebastien Bougleux (Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC, Caen, France)
 */
/* -----------------------------------------------------------   
   This file is part of LSAPE.
   
   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.
   
   You should have received a copy of the CeCILL-C License along with 
   LSAPE. If not, see http://www.cecill.info/index.en.html

   -----------------------------------------------------------   
   Creation: December 5 2015
   Modifications: March 2018
   
   -----------------------------------------------------------   
   Use function greedyLSAP in lsap.h for top-level usage
*/
// =========================================================================

#ifndef _GREEDY_LSAP_H_
#define _GREEDY_LSAP_H_

#include <iostream>
#include <limits>
#include <algorithm>
#include "deftypes.h"
#include "greedy-cost-sort.h"
#include "utils.h"

namespace lsape {

  // -----------------------------------------------------------
  /**
   * \brief Assignment with basic greedy algorithm
   * \param[in] C Cost matrix represented as an array of size nrows*ncols
   * \param[in] nrows Number of rows (size of the 1st set)
   * \param[in] ncols Numver of columns (size of the 2nd set)
   * \param[out] rho Assignement from the 1st set to the 2nd one
   * \param[out] varrho Assignment from the 2nd set to the 1st one (optional)
   * \param[in] permS Permutation of the smallest set of indicies (optional, identity by default)
   * \param[in] permL Permutation of the largest set of indicies (optional, identity by default)
   * \return The cost of the assignement with respect to \p C
   * \note Indicies of unassigned elements are stored in arrays. Each iteration assigns an unassigned element i of the smallest set to an unassigned element j of the largest set so that the cost C_{i,j} is minimal. 
   */
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyBasicLSAP(const DT *C, const IT &nrows, const IT &ncols, IT *rho, IT *varrho = NULL, IT *permS = NULL, IT *permL = NULL)
  {
    IT i, imin, itmp;
    DT cmin, mxdt = std::numeric_limits<DT>::max(), approx = 0;
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[ncols]; }
    IT nmx = std::max(nrows,ncols);

    IT *unass = new IT[nmx+1], *pti_unass = NULL;
    IT *pti_unass_beg = unass, *pti_unass_end = unass+nmx, *pti_min = NULL;
  
    if (nrows >= ncols) // assign columns
      {
	if (permL == NULL) for (i = 0; i < nrows; i++) { unass[i] = i; rho[i] = ncols; }
	else for (i = 0; i < nrows; i++) { unass[i] = permL[i]; rho[i] = ncols; }
    
	if (permS == NULL)
	  {
	    for (IT j = 0; j < ncols; j++)
	      {
		// find the min among unassigned rows
		cmin = mxdt;
		for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
		  {
		    const DT &cij = C[j*nrows+*pti_unass];
		    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
		  }
		// assign the row which provides the minimum and update the approximate solution
		imin = *pti_min; rho[imin] = j; varrho[j] = imin;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		approx += cmin;
	      }
	  }
	else // a permutation of the rows is given
	  {
	    for (IT jj = 0, *permit = permS, j = 0; jj < ncols; jj++, ++permit)
	      {
		j = *permit;
		// find the min among unassigned rows
		cmin = mxdt;
		for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
		  {
		    const DT &cij = C[j*nrows+*pti_unass];
		    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
		  }
		// assign the row which provides the minimum and update the approximate solution
		imin = *pti_min; rho[imin] = j; varrho[j] = imin;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		approx += cmin;
	      }
	  }
      }
    else // assign rows
      {
	if (permL == NULL) for (i = 0; i < ncols; i++) { unass[i] = i; varrho[i] = nrows; }
	else for (i = 0; i < ncols; i++) { unass[i] = permL[i]; varrho[i] = nrows; }
    
	if (permS == NULL)
	  {  
	    for (i = 0; i < nrows; i++)
	      {
		// find the min among unassigned columns
		cmin = mxdt;
		for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
		  {
		    const DT &cij = C[*pti_unass * nrows + i];
		    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
		  }
		// assign the column which provides the minimum and update the approximate solution
		imin = *pti_min; varrho[imin] = i; rho[i] = imin;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		approx += cmin;
	      }
	  }
	else // a permutation of the rows is given
	  {
	    for (IT ii = 0, *permit = permS; ii < nrows; ii++, ++permit)
	      {
		i = *permit;
		// find the min among unassigned columns
		cmin = mxdt;
		for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
		  {
		    const DT &cij = C[*pti_unass * nrows + i];
		    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
		  }
		// assign the column which provides the minimum and update the approximate solution
		imin = *pti_min; varrho[imin] = i; rho[i] = imin;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		approx += cmin;
	      }
	  }
      }
    delete[] unass;
    if (deletevarrho) delete[] varrho;
    return approx;
  }

  // -----------------------------------------------------------
  // Refined greedy LSAP for n >= m
  // return the cost of the approximate solution
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyRefinedLSAPcols(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL, IT *permS = NULL, IT *permL = NULL)
  {
    IT nass = 0, i, j, imin, jmin;
    DT cmin, ckmin, mxdt = std::numeric_limits<DT>::max(), approx = 0;
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[m]; }

    IT *unassi = new IT[n+1], *pti_unass = NULL;
    IT *pti_unass_beg = unassi, *pti_unass_end = unassi+n, *pti_min = NULL;

    IT *ptj_unass = NULL, *unassj = new IT[m+1];
    IT *ptj_unass_beg = unassj, *ptj_unass_end = unassj+m, *ptj_min = NULL;

    IT *ptj_unass1 = NULL;
  
    if (permL == NULL) for (i = 0; i < n; i++) { unassi[i] = i; rho[i] = m; }
    else for (i = 0; i < n; i++) { unassi[i] = permL[i]; rho[i] = m; }
    if (permS == NULL) for (j = 0; j < m; j++) { unassj[j] = j; }
    else for (j = 0; j < m; j++) { unassj[j] = permS[j]; }
  
    // augmentation of columns
    for (ptj_unass1 = ptj_unass_beg; ptj_unass1 != ptj_unass_end;)
      {
	j = *ptj_unass1;
	// find the min among unassigned rows
	cmin = mxdt;
	for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
	  {
	    const DT &cij = C[j*n+*pti_unass];
	    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
	  }
	// find the min among unassigned columns for imin
	imin = *pti_min;
	ckmin = mxdt;
	for (ptj_unass = ptj_unass_beg; ptj_unass != ptj_unass_end; ptj_unass++)
	  {
	    const DT &cik = C[*ptj_unass*n+imin];
	    if (cik  < ckmin) { ckmin = cik; ptj_min = ptj_unass; }
	  }
	// assign the row and column which provides the minimum
	if (cmin <= ckmin)
	  {
	    rho[imin] = j; varrho[j] = imin;
	    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
	    ptj_unass_beg++;
	    approx += cmin;
	  }
	else
	  {
	    jmin = *ptj_min; rho[imin] = jmin; varrho[jmin] = imin;
	    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
	    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
	    approx += ckmin;
	  }
	ptj_unass1 = ptj_unass_beg;
      }
  
    delete[] unassi; delete[] unassj;
    if (deletevarrho) delete[] varrho;
    return approx;
  }

  // -----------------------------------------------------------
  // Refined greedy LSAP for n <= m
  // return the cost of the approximate solution
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyRefinedLSAProws(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL, IT *permS = NULL, IT *permL = NULL)
  {
    IT nass = 0, i, j, imin, jmin;
    DT cmin, ckmin, mxdt = std::numeric_limits<DT>::max(), approx = 0;
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[m]; }

    IT *unassi = new IT[n+1], *pti_unass = NULL;
    IT *pti_unass_beg = unassi, *pti_unass_end = unassi+n, *pti_min = NULL;

    IT *ptj_unass = NULL, *unassj = new IT[m+1];
    IT *ptj_unass_beg = unassj, *ptj_unass_end = unassj+m, *ptj_min = NULL;

    IT *pti_unass1 = NULL;
  
    if (permS == NULL) for (i = 0; i < n; i++) { unassi[i] = i; }
    else for (i = 0; i < n; i++) { unassi[i] = permS[i]; }
    if (permL == NULL) for (j = 0; j < m; j++) { unassj[j] = j; varrho[j] = n; }
    else for (j = 0; j < m; j++) { unassj[j] = permL[j]; varrho[j] = n; }
  
    // augmentation of rows
    for (pti_unass1 = pti_unass_beg; pti_unass1 != pti_unass_end;)
      {
	i = *pti_unass1;
	// find the min among unassigned columns
	cmin = mxdt;
	for (ptj_unass = ptj_unass_beg; ptj_unass != ptj_unass_end; ptj_unass++)
	  {
	    const DT &cij = C[*ptj_unass*n+i];
	    if (cij  < cmin) { cmin = cij; ptj_min = ptj_unass; }
	  }
	// find the min among unassigned rows for jmin
	jmin = *ptj_min;
	ckmin = mxdt;
	for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
	  {
	    const DT &cik = C[jmin*n+*pti_unass];
	    if (cik  < ckmin) { ckmin = cik; pti_min = pti_unass; }
	  }
	// assign the row and column which provides the minimum
	if (cmin <= ckmin)
	  {
	    varrho[jmin] = i; rho[i] = jmin;
	    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
	    pti_unass_beg++; 
	    approx += cmin;
	  }
	else
	  {
	    imin = *pti_min; varrho[jmin] = imin; rho[imin] = jmin;
	    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
	    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
	    approx += ckmin;
	  }
	pti_unass1 = pti_unass_beg;
      }
  
    delete[] unassi; delete[] unassj;
    if (deletevarrho) delete[] varrho;
    return approx;
  }

  // -----------------------------------------------------------
  /**
   * \brief Assignment with refined basic greedy algorithm
   * \param[in] C Cost matrix represented as an array of size nrows*ncols
   * \param[in] nrows Number of rows (size of the 1st set)
   * \param[in] ncols Numver of columns (size of the 2nd set)
   * \param[out] rho Assignement from the 1st set to the 2nd one
   * \param[out] varrho Assignment from the 2nd set to the 1st one (optional)
   * \param[in] permS Permutation of the smallest set of indicies (optional, identity by default)
   * \param[in] permL Permutation of the largest set of indicies (optional, identity by default)
   * \return The cost of the assignement with respect to \p C
   */
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyRefinedLSAP(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL, IT *permS = NULL, IT *permL = NULL)
  {
    if (n >= m) return greedyRefinedLSAPcols(C,n,m,rho,varrho,permS,permL);
    return greedyRefinedLSAProws(C,n,m,rho,varrho,permS,permL);
  }

  // -----------------------------------------------------------
  // Loss greedy LSAP for n >= m
  // return the cost of the approximate solution
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyLossLSAPcols(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL)
  {
    IT nass = 0, i, j, imin, jmin, imin2, imin3;
    DT cmin, cij, ckmin, mxdt = std::numeric_limits<DT>::max(), approx = 0, cmin2, cmin3;
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[m]; }

    IT *unassi = new IT[n+1], *pti_unass = NULL;
    IT *pti_unass_beg = unassi, *pti_unass_end = unassi+n, *pti_min = NULL, *pti_min2 = NULL, *pti_min3 = NULL;

    IT *ptj_unass = NULL, *unassj = new IT[m+1];
    IT *ptj_unass_beg = unassj, *ptj_unass_end = unassj+m, *ptj_min = NULL;

    IT *ptj_unass1 = NULL;
  
    for (i = 0; i < n; i++) { unassi[i] = i; rho[i] = m; }
    for (j = 0; j < m; j++) { unassj[j] = j; }
  
    // augmentation of columns
    for (ptj_unass1 = ptj_unass_beg; ptj_unass1 != ptj_unass_end;)
      {
	j = *ptj_unass1;
	// find the min among unassigned rows
	cmin = mxdt;
	for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
	  {
	    cij = C[j*n+*pti_unass];
	    if (cij  < cmin) { cmin = cij; pti_min = pti_unass; }
	  }
	// find the min among unassigned columns for imin
	imin = *pti_min;
	ckmin = mxdt;
	for (ptj_unass = ptj_unass_beg; ptj_unass != ptj_unass_end; ptj_unass++)
	  {
	    const DT &cik = C[*ptj_unass*n+imin];
	    if (cik  < ckmin) { ckmin = cik; ptj_min = ptj_unass; }
	  }
	// assign the row and column which provides the minimum
	jmin = *ptj_min;
	if (j == jmin)
	  {
	    rho[imin] = j; varrho[j] = imin;
	    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
	    ptj_unass_beg++;
	    approx += cmin;
	  }
	else
	  {
	    // find the min among unassigned rows different to imin, for j => find the 2nd min
	    cmin3 = cmin2 = mxdt;
	    for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
	      {
		if (*pti_unass != imin)
		  {
		    cij = C[j*n+*pti_unass];
		    if (cij  < cmin2) { cmin2 = cij; pti_min2 = pti_unass; }
		    cij = C[jmin*n+*pti_unass];
		    if (cij  < cmin3) { cmin3 = cij; pti_min3 = pti_unass; }
		  }
	      }
	    imin2 = *pti_min2;
	    imin3 = *pti_min3;
	    if (cmin + cmin3 < cmin2 + ckmin) // remove j, jmin, imin, imin3
	      {
		rho[imin] = j; varrho[j] = imin;
		rho[imin3] = jmin; varrho[jmin] = imin3;
		ptj_unass_beg++;
		*ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		if (imin3 == *pti_unass_beg)
		  {
		    pti_unass_beg++;
		    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		  }
		else {
		  *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		  *pti_min3 = *pti_unass_beg; *pti_unass_beg = imin3; pti_unass_beg++;
		}
		approx += cmin + cmin3;
	      }
	    else // remove j, jmin, imin, imin2
	      {
		rho[imin2] = j; varrho[j] = imin2;
		rho[imin] = jmin; varrho[jmin] = imin;
		ptj_unass_beg++;
		*ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		if (imin2 == *pti_unass_beg)
		  {
		    pti_unass_beg++;
		    *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		  }
		else {
		  *pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		  *pti_min2 = *pti_unass_beg; *pti_unass_beg = imin2; pti_unass_beg++;  
		}
		approx += ckmin + cmin2;
	      }
	  }
	ptj_unass1 = ptj_unass_beg;
      }
  
    delete[] unassi; delete[] unassj;
    if (deletevarrho) delete[] varrho;
    return approx;
  }

  // -----------------------------------------------------------
  // Loss greedy LSAP for n <= m
  // return the cost of the approximate solution
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyLossLSAProws(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL)
  {
    IT nass = 0, i, j, imin, jmin, jmin2, jmin3;
    DT cmin, cij, ckmin, mxdt = std::numeric_limits<DT>::max(), approx = 0, cmin2, cmin3;
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[m]; }

    IT *unassi = new IT[n+1], *pti_unass = NULL;
    IT *pti_unass_beg = unassi, *pti_unass_end = unassi+n, *pti_min = NULL;

    IT *ptj_unass = NULL, *unassj = new IT[m+1];
    IT *ptj_unass_beg = unassj, *ptj_unass_end = unassj+m, *ptj_min = NULL, *ptj_min2 = NULL, *ptj_min3 = NULL;

    IT *pti_unass1 = NULL;
  
    for (i = 0; i < n; i++) { unassi[i] = i; }
    for (j = 0; j < m; j++) { unassj[j] = j; varrho[j] = n; }
  
    // augmentation of rows
    for (pti_unass1 = pti_unass_beg; pti_unass1 != pti_unass_end;)
      {
	i = *pti_unass1;
	// find the min among unassigned columns
	cmin = mxdt;
	for (ptj_unass = ptj_unass_beg; ptj_unass != ptj_unass_end; ptj_unass++)
	  {
	    cij = C[*ptj_unass*n+i];
	    if (cij  < cmin) { cmin = cij; ptj_min = ptj_unass; }
	  }
	// find the min among unassigned rows for jmin
	jmin = *ptj_min;
	ckmin = mxdt;
	for (pti_unass = pti_unass_beg; pti_unass != pti_unass_end; pti_unass++)
	  {
	    const DT &cik = C[jmin*n+*pti_unass];
	    if (cik  < ckmin) { ckmin = cik; pti_min = pti_unass; }
	  }
	// assign the row and column which provides the minimum
	imin = *pti_min;
	if (i == imin)
	  {
	    varrho[jmin] = i; rho[i] = jmin;
	    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
	    pti_unass_beg++;
	    approx += cmin;
	  }
	else
	  {
	    // find the min among unassigned columns different to jmin, for i => find the 2nd min
	    cmin3 = cmin2 = mxdt;
	    for (ptj_unass = ptj_unass_beg; ptj_unass != ptj_unass_end; ptj_unass++)
	      {
		if (*ptj_unass != jmin)
		  {
		    cij = C[*ptj_unass*n+i];
		    if (cij  < cmin2) { cmin2 = cij; ptj_min2 = ptj_unass; }
		    cij = C[*ptj_unass*n+imin];
		    if (cij  < cmin3) { cmin3 = cij; ptj_min3 = ptj_unass; }
		  }
	      }
	    jmin2 = *ptj_min2;
	    jmin3 = *ptj_min3;
	    if (cmin + cmin3 < cmin2 + ckmin) // remove i, imin, jmin, jmin3
	      {
		varrho[jmin] = i; rho[i] = jmin;
		varrho[jmin3] = imin; rho[imin] = jmin3;
		pti_unass_beg++;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		if (jmin3 == *ptj_unass_beg)
		  {
		    ptj_unass_beg++;
		    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		  }
		else {
		  *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		  *ptj_min3 = *ptj_unass_beg; *ptj_unass_beg = jmin3; ptj_unass_beg++;
		}
		approx += cmin + cmin3;
	      }
	    else // remove i, imin, jmin, jmin2
	      {
		varrho[jmin2] = i; rho[i] = jmin2;
		varrho[jmin] = imin; rho[imin] = jmin;
		pti_unass_beg++;
		*pti_min = *pti_unass_beg; *pti_unass_beg = imin; pti_unass_beg++;
		if (jmin2 == *ptj_unass_beg)
		  {
		    ptj_unass_beg++;
		    *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		  }
		else {
		  *ptj_min = *ptj_unass_beg; *ptj_unass_beg = jmin; ptj_unass_beg++;
		  *ptj_min2 = *ptj_unass_beg; *ptj_unass_beg = jmin2; ptj_unass_beg++;  
		}
		approx += ckmin + cmin2;
	      }
	  }
	pti_unass1 = pti_unass_beg;
      }
  
    delete[] unassi; delete[] unassj;
    if (deletevarrho) delete[] varrho;
    return approx;
  }

  // -----------------------------------------------------------
  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyLossLSAP(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL)
  {
    if (n >= m) return greedyLossLSAPcols(C,n,m,rho,varrho);
    return greedyLossLSAProws(C,n,m,rho,varrho);
  }

  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT greedyBasicSortLSAP(const DT *C, const IT &n, const IT &m, IT *rho, IT *varrho = NULL)
  {
    // sort the costs
    IT *idxs = new IT[n*m+1], nmn = std::min(n,m);
    IT *pt_idxs_end = idxs+n*m, i = 0, j;
    DT approx = 0;
    for (IT *pt_idx = idxs; pt_idx != pt_idxs_end; pt_idx++, i++) *pt_idx = i;

    // OTHER SORTING FCT MAY BE USED
    CostSortComp<DT,IT> comp(C);
    std::sort(idxs,idxs+n*m,comp);

    // assign element in ascending order of the costs
    bool deletevarrho = false;
    if (varrho == NULL) { deletevarrho = true; varrho = new IT[m]; }
    for (IT i = 0; i < n; i++) rho[i] = m;
    for (IT j = 0; j < m; j++) varrho[j] = n;
    for (IT *pt_idx = idxs, nbe = 0; pt_idx != pt_idxs_end && nbe < nmn; pt_idx++)
      {
	ind2sub(*pt_idx,n,i,j);
	if (rho[i] == m && varrho[j] == n)
	  {
	    rho[i] = j; varrho[j] = i;
	    approx += C[*pt_idx];
	    nbe++;
	  }
      }
    if (deletevarrho) delete[] varrho;
    delete[] idxs;
    return approx;
  }

  // -----------------------------------------------------------
  /**
   * \brief Counting sort greedy LSAP with integer cost values
   * \return the cost of the assignment
   */
  // -----------------------------------------------------------
  template <class DT>
    DT greedyBasicCountingSortLSAP(const DT *C, const LSAPE_IndexType &n, const LSAPE_IndexType &m, LSAPE_IndexType *rho, LSAPE_IndexType *varrho = NULL)
  {
    DT approx = 0;
    if (!std::numeric_limits<DT>::is_integer) throw std::runtime_error("DT template parameter must be an integer type for greedyBasicCountingSortLSAP<DT>(...).");
    else
    {
      // find min and max values
      DT minc = C[0], maxc = C[0], nmn = std::min(n,m);
      const DT *ite = C+n*m;
      for (const DT *itc = C+1; itc < ite; itc++)
	{
	  const DT &vc = *itc;
	  if (vc < minc) minc = vc;
	  else if (vc > maxc) maxc = vc;
	}
  
      // construct histogram
      LSAPE_IndexType nbins = maxc - minc + 1;
      LSAPE_IndexType *bins = new LSAPE_IndexType[nbins];
      const LSAPE_IndexType *itbe = bins+nbins;
      for (LSAPE_IndexType *itb = bins; itb < itbe; itb++) *itb = 0;
      for (const DT *itc = C; itc < ite; itc++) bins[static_cast<LSAPE_IndexType>(*itc-minc)]++;

      // starting index for each cost value
      LSAPE_IndexType tot = 0, oldcount;
      for (LSAPE_IndexType i = 0; i < nbins; i++) { oldcount = bins[i]; bins[i] = tot; tot += oldcount; }

      // reoder the costs, preserving order of C with equal keys
      LSAPE_IndexType *idxs = new LSAPE_IndexType[n*m+1], k = 0;
      for (const DT *itc = C; itc < ite; itc++, k++) { idxs[bins[static_cast<LSAPE_IndexType>(*itc-minc)]] = k; bins[static_cast<LSAPE_IndexType>(*itc-minc)]++; }

      // assign element in ascending order of the costs
      LSAPE_IndexType *pt_idxs_end = idxs+n*m, i = 0, j;
      bool deletevarrho = false;
      if (varrho == NULL) { deletevarrho = true; varrho = new LSAPE_IndexType[m]; }
      for (LSAPE_IndexType i = 0; i < n; i++) rho[i] = m;
      for (LSAPE_IndexType j = 0; j < m; j++) varrho[j] = n;
      for (LSAPE_IndexType *pt_idx = idxs, nbe = 0; pt_idx != pt_idxs_end && nbe < nmn; pt_idx++)
	{
	  ind2sub(*pt_idx,n,i,j);
	  if (rho[i] == m && varrho[j] == n)
	    {
	      rho[i] = j; varrho[j] = i;
	      approx += C[*pt_idx];
	      nbe++;
	    }
	}
      if (deletevarrho) delete[] varrho;
      delete[] idxs; delete[] bins;
    }
    return approx;
  }
  
} // end namespace lsape
  
// -----------------------------------------------------------
#endif
