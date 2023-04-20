// -----------------------------------------------------------   
/** \file lsap.h
 *  \brief Hungarian algorithm for Linear Sum Assignment Problem (LSAP) and its dual problem, for balanced and unbalanced instances.
 *  \author Sebastien Bougleux (Normandie Univ, UNICAEN, ENSICAEN, CNRS, GREYC, Image team, Caen, France)
*/
/* -----------------------------------------------------------
   This file is part of LSAPE.
   
   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.

   -----------------------------------------------------------   
   Creation: December 5 2015
   Last modif: March 2018
*/

#ifndef __LSAP_H_
#define __LSAP_H_

#include <cstring>
#include <limits>
#include "dev/deftypes.h"
#include "dev/hungarian-lsap.h"
#include "dev/greedy-lsap.h"
#include "dev/enum-matchings.h"

namespace lsape {

  // -----------------------------------------------------------
  /*template <class DT, typename IT>
    DT min(const DT *C, const IT &nrows, const IT &ncols)
  {
    DT cmin = std::numeric_limits<DT>::max();
    const DT *cit = *C, cend = *(C+nrows*ncols);
    for (; cit != cend; ++cit) if (*cit < cmin) cmin = *cit;
    return cmin;
  }

  // -----------------------------------------------------------
  template <class DT, typename IT>
    void sum(const DT &c, DT *C, const IT &nrows, const IT &ncols)
  {
    DT *cit = *C, cend = *(C+nrows*ncols);
    for (; cit != cend; ++cit) *cit += c;
    }*/

  // -----------------------------------------------------------
  template <class DT, typename IT>
    DT* unbalanced2balanced(const DT *C, const IT &nrows, const IT &ncols)
  {
    const IT mx = std::max(nrows,ncols);
    DT *balC = new DT[mx*mx];

    if (nrows < ncols)
      {
	const IT d = mx-nrows;
	for (IT j = 0; j < ncols; j++)
	  {
	    std::memcpy(balC+mx*j,C+nrows*j,sizeof(DT)*nrows);
	    std::memset(balC+mx*j+nrows,0,sizeof(DT)*d);
	  }
      }
    else
      {
	std::memcpy(balC,C,sizeof(DT)*nrows*ncols);
	std::memset(balC+nrows*ncols,0,sizeof(DT)*nrows*(mx-ncols));
      }
  
    return balC;
  }

  // -----------------------------------------------------------
  /**
   * \brief Cost of an assignment, given dual variables
   * \param[in] u Dual variables for the 1st set
   * \param[in] nr Size of the 1st set
   * \param[in] v Dual variables for the 2nd set
   * \param[in] nc Size of the 2nd set
   * \return Cost of the assignment
   */
  template <typename DT, typename IT>
    DT permCost(const DT *u, const IT &nr, const DT *v, const IT &nc)
  {
    DT sumcost = 0;
    for (int i = 0; i < nr; i++) sumcost += u[i];
    for (int j = 0; j < nc; j++) sumcost += v[j];
    return sumcost;
  }

  // -----------------------------------------------------------
  template <typename DT, typename IT>
    DT permCost(const DT *C, const IT &nr, const IT *rho)
  {
    DT sumcost = 0;
    for (int i = 0; i < nr; i++) if (rho[i] >= 0) sumcost += C[rho[i]*nr+i];
    return sumcost;
  }

  // --------------------------------------------------------------------------------
  /**
   * \brief Compute a solution to symmetric or assymetric LSAP with the Hungarian algorithm
   * \param[in] C nxm cost matrix represented as an array of size \c nm obtained by concatenating its columns
   * \param[in] nrows Number of rows of \p C (size of the 1st set)
   * \param[in] cols Number of columns of \p C (size of the 2nd set)
   * \param[out] rho A solution to the LSAP: an array of size \p n (must be previously allocated), rho[i]=-1 indicates that i is not assigned, else rho[i]=j indicates that i is assigned to j
   * \param[out] u Array of dual variables associated to the 1st set (rows of \p C)
   * \param[out] v Array of dual variables associated to the 2nd set (columns of \p C)
   * \param[in] init_type 0: no initialization, 1: classical (default)
   * \param[in] forb_assign If true, forbidden assignments are marked with negative values in the cost matrix
   * \details A solution to the LSAP is computed with the primal-dual version of the Hungarian algorithm, as detailed in:
   * \li <em>R. Burkard, M. Dell'Amico and S. Martello. Assignment Problems. SIAM, 2009</em>
   * \li <em>E.L. Lawler. Combinatorial Optimization: Networks and Matroids. Holt, Rinehart and Winston, New York, 1976</em>
   *
   * This version updates dual variables \c u and \c v instead of the cost matrix \c C, and at each iteration, the current matching is augmented by growing only one Hungarian tree until an augmenting path is found. Our implementation uses a Bread-First-like strategy to construct the tree, according to a FIFO strategy to select the next element at each iteration of the growing.
   *
   * Complexities:
   * \li O(min{n,m}²max{n,m}) in time (worst-case)
   * \li O(nm) in space
   *
   * \remark
   * Template \p DT allows to compute a solution with integer or floating-point values. Note that rounding errors may occur with floating point values when dual variables are updated but this does not affect the overall process. These errors can be observed when the reduced cost matrix is computed from the dual variables.
   */
  // --------------------------------------------------------------------------------
  template <class DT>
    void hungarianLSAP(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, LSAPE_IndexType *rho, DT *u, DT *v,
		       LSAPE_IndexType *varrho = NULL, unsigned short init_type = 1, bool forb_assign = false)
  {
    if (nrows == ncols) hungarianSquareLSAP<DT,LSAPE_IndexType>(C,nrows,rho,u,v,varrho,init_type,forb_assign);
    else hungarianRectLSAP<DT,LSAPE_IndexType>(C,nrows,ncols,rho,varrho,u,v,init_type,forb_assign);
  }

  // --------------------------------------------------------------------------------
  /**
   * @brief Approximative low-cost solution to an LSAP instance, with greedy algorithms
   * @param[in] C nxm cost matrix, represented as an array of size \c nm obtained by concatenating its columns
   * @param[in] nrows Number of rows of \p C
   * @param[in] ncols Number of columns of \p C
   * @param[out] rho Array of size \p n (must be previously allocated) for the assignment of the rows to the columns, rho[i]=-1 w if i is not assigned, else rho[i]=j
   * @param[out] varrho Optional array of size \p m (must be previously allocated) for the assignement of the columns to the rows
   * @param[in] greedy_type 0:Basic, 1:Refined, 2:Loss (default), 3:Basic sort, 4:Basic counting sort (integers only)
   * @param[in] a permutation of the smallest set between rows and columns
   * @param[in] a permutation of the largest set between rows and columns
   * @return Cost of the approximative solution
   */
  // --------------------------------------------------------------------------------
  template <class DT>
    DT greedyLSAP(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, LSAPE_IndexType *rho,
		  LSAPE_IndexType *varrho = NULL, unsigned short greedy_type = 2, LSAPE_IndexType *permS = NULL, LSAPE_IndexType *permL = NULL)
  {
    switch(greedy_type)
      {
      case 0: return greedyBasicLSAP<DT,LSAPE_IndexType>(C,nrows,ncols,rho,varrho,permS,permL);
      case 1: return greedyRefinedLSAP<DT,LSAPE_IndexType>(C,nrows,ncols,rho,varrho,permS,permL);
      case 3: return greedyBasicSortLSAP<DT,LSAPE_IndexType>(C,nrows,ncols,rho,varrho);
      case 4: return greedyBasicCountingSortLSAP<DT>(C,nrows,ncols,rho,varrho);
      default: return greedyLossLSAP<DT,LSAPE_IndexType>(C,nrows,ncols,rho,varrho);
      }
  }

  // --------------------------------------------------------------------------------
  /**
   * @brief Enumerate solutions to an LSAP instance
   * \param[in] C nxm cost matrix
   * \param[in] nrows Number of rows of \p C (size of the 1st set)
   * \param[in] ncols Number of columns of \p C (size of the 2nd set)
   * \param[out] nsol Number of solutions found
   * \param[out] solutions Array of solutions
   */
  // --------------------------------------------------------------------------------
  template <typename DT>
    void lsapSolutions(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const int &ksol, LSAPE_IndexType *r2c, DT *u, DT *v, std::list<LSAPE_IndexType*> &solutions)
  {
    // construct equality directed graph
    cDigraph<LSAPE_IndexType> edg = equalityDigraph<DT,LSAPE_IndexType>(C,nrows,ncols,r2c,u,v);

    // enuerate k solutions, at most
    AllPerfectMatchings<LSAPE_IndexType> apm(edg,solutions);
    apm.enumPerfectMatchings(edg,ksol);
  }

  // -----------------------------------------------------------
  inline void lmatchings2mmachings(const std::list<LSAPE_IndexType*> &lmatchings, LSAPE_IndexType *mmatchings, const LSAPE_IndexType &nrows, LSAPE_IndexType scal = 0)
  {
    typename std::list<LSAPE_IndexType*>::const_iterator it = lmatchings.begin();
    int nsol = lmatchings.size();

    for (int s = 0; s < nsol; s++, ++it)
    {
      const LSAPE_IndexType *tab = *it;
      for (int i = 0; i < (int)(nrows); i++) mmatchings[s*nrows+i] = tab[i] + scal;
    }
  }

  // -----------------------------------------------------------
  inline void lmatchingsFree(std::list<LSAPE_IndexType*> &l)
  {
    for(typename std::list<LSAPE_IndexType*>::iterator it = l.begin(); it != l.end(); it++) { delete[] *it; *it = NULL; }
    l.clear();
  }
  
  // -----------------------------------------------------------
  /**
   * \brief Enumerate solutions to an LSAP instance
   * \param[in] C nxm cost matrix
   * \param[in] nrows Number of rows of \p C (size of the 1st set)
   * \param[in] ncols Number of columns of \p C (size of the 2nd set)
   * \param[out] nsol Number of solutions found
   * \param[out] solutions Array of solutions
   * @param[in] ksol Maximum number of solutions to enumerate
   * @return Minimal cost associated to the solutions
   */
  template <typename DT>
    DT lsapSolutions(const DT *C, const LSAPE_IndexType &nrows, const LSAPE_IndexType &ncols, const int &ksol, std::list<LSAPE_IndexType*> &solutions)
  {
    if (nrows != ncols) throw std::runtime_error("nrows must be equal to ncols for lsapSolutions(...)");
    LSAPE_IndexType *r2c = new LSAPE_IndexType[nrows];
    DT *u = new DT[nrows], *v = new DT[ncols], mCost = 0;
   
    // find a solution
    hungarianLSAP<DT>(C,nrows,ncols,r2c,u,v);
    for (LSAPE_IndexType i = 0; i < nrows; i++) mCost += u[i];
    for (LSAPE_IndexType j = 0; j < ncols; j++) mCost += v[j];
    
    // enumerate solutions
    lsapSolutions<DT>(C,nrows,ncols,ksol,r2c,u,v,solutions);

    delete[] r2c; delete[] u; delete[] v;

    return mCost;
  }

} // namespace
// -----------------------------------------------------------
#endif
