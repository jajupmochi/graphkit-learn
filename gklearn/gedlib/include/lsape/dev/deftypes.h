// -----------------------------------------------------------   
/** \file deftypes.h
    \brief Types for matchings and ec-matchings
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

#ifndef _DEFTYPES_H_
#define _DEFTYPES_H_

#ifndef LSAPE_IndexType
#define LSAPE_IndexType unsigned int
#endif

namespace lsape {
  
  enum GREEDY_METHOD { BASIC=0, REFINED=1, LOSS=2, BASIC_SORT=3, INT_BASIC_SORT=4 };
  
  enum LSAPE_MODEL { ECBP=0, FLWC=1, EBP=2, FLCC=3, FBP=4, FBP0=5, SFBP=6 };

} // end namespace

#endif
