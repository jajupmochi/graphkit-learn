// -----------------------------------------------------------   
/** \file greedy-sort-cost.h
    \brief Sorting strategy for greedy sort LSAP/LSAPE
    \author Sebastien Bougleux (Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC, Caen, France)
   -----------------------------------------------------------
   
   This file is part of LSAPE.
   
   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.
   -----------------------------------------------------------
   
   Creation: March 2018
   Last modif: 
*/

#ifndef _GREEDY_COST_SORT_H_
#define _GREEDY_COST_SORT_H_

namespace lsape {
  
  // -----------------------------------------------------------
  // BasicSort greedy LSAP/LSAPE for n >= m only (not checked)
  // return the cost of the approximate solution
  // -----------------------------------------------------------
  template <class DT, typename IT>
    class CostSortComp
  {
  private:
    const DT *_C;
  public:
  CostSortComp(const DT *C) : _C(C) { }
    ~CostSortComp() { }
    inline bool operator()(const IT &ij, const IT &kl) { return (_C[ij] < _C[kl]); }
  };

}

#endif
