/**
 * @file JacobiAuctionLSAP.h
 * @author Évariste <<evariste.daller@unicaen.fr>>
 * @version Apr 12 2017
 */

#ifndef __JACOBIAUCTIONLSAP_H__
#define __JACOBIAUCTIONLSAP_H__


#include "AuctionLSAP.h"


/**
 * @class JacobiAuctionLSAP
 * @author evariste
 * @date 14/04/17
 * @file JacobiAuctionLSAP.h
 * @brief Jacobi version of the auction algorithm
 */
template <typename DT, typename IT>
class JacobiAuctionLSAP :
    public AuctionLSAP<DT,IT>
{

protected:

  /**
   * @brief All the unassigned vertex in $G_1$ are returned
   */
  virtual std::list<IT> chooseBidders() const {
    std::list<IT> bidders;
    for (IT i=0; i<this->_n; i++){
      if (this->_unassigned[i])   bidders.emplace_back(i);
    }
    return bidders;
  }

public:

  JacobiAuctionLSAP( const double& firstEpsilon,
                     const int& n ):
      AuctionLSAP<DT,IT>(firstEpsilon, n)
  {}

  JacobiAuctionLSAP( const double& firstEpsilon,
                     const double& finalEpsilon ):
      AuctionLSAP<DT,IT>(firstEpsilon, finalEpsilon)
  {}
  
  JacobiAuctionLSAP( const DT* C,
                     const double& finalEpsilon ):
      AuctionLSAP<DT,IT>(C, finalEpsilon)
  {}

};


#endif
