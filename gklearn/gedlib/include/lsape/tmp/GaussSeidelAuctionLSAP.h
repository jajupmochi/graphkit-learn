/**
 * @file GaussSeidelAuctionLSAP.h
 * @author Évariste <<evariste.daller@unicaen.fr>>
 * @version Apr 13 2017
 */

#ifndef __GAUSSSEIDELAUCTIONLSAP_H__
#define __GAUSSSEIDELAUCTIONLSAP_H__


#include "AuctionLSAP.h"


/**
 * @class GaussSeidelAuctionLSAP
 * @author evariste
 * @date 14/04/17
 * @file GaussSeidelAuctionLSAP.h
 * @brief Gauss-Seidel version of the auction algorithm
 */
template <typename DT, typename IT>
class GaussSeidelAuctionLSAP :
    public AuctionLSAP<DT,IT>
{

protected:

  /**
   * @brief The first vertex in $G_1$ that is unassigned is returned
   * @return A list containing a unique element
   */
  virtual std::list<IT> chooseBidders() const {
    std::list<IT> bidders;
    int i=0;
    while ( i<this->_n && !(this->_unassigned[i]) ) i++;
    if (i<this->_n)   bidders.emplace_back(i);
    
    return bidders;
  }
  

public:

  GaussSeidelAuctionLSAP( const double& firstEpsilon,
                     const int& n ):
      AuctionLSAP<DT,IT>(firstEpsilon, n)
  {}

  GaussSeidelAuctionLSAP( const double& firstEpsilon,
                     const double& finalEpsilon ):
      AuctionLSAP<DT,IT>(firstEpsilon, finalEpsilon)
  {}
  
  GaussSeidelAuctionLSAP( const DT* C,
                          const double& finalEpsilon ):
      AuctionLSAP<DT,IT>(C, finalEpsilon)
  {}
  

};


#endif
