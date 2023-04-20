/**
 * @file AuctionLSAP.h
 * @author Évariste <<evariste.daller@unicaen.fr>>
 * @version Apr 12 2017
 */

#ifndef __AUCTIONLSAP_H__
#define __AUCTIONLSAP_H__


#include <list>
#include "utils.hh"


/**
 * @class AuctionLSAP
 * @author evariste
 * @date 12/04/17
 * @file AuctionLSAP.h
 * @brief Base class of sequential auction algorithms
 */
template <typename DT, typename IT>
class AuctionLSAP {

protected: /* MEMBERS */

  double _epsilon;        //!< Current minimum bid
  double _finalEpsilon;   //!< Lower bound of epsilon to stop eps-scaling
  double _scalingFactor;  //!< Scaling factor for epsilon scaling phase

  bool*  _unassigned;     //!< <code>_unassigned[i]</code> is true if vertex i in $G_1$ is not mapped
  IT     _nbUnassigned;   //!< Number of $i$ such that <code>_unassigned[i] == true</code>

  double* _bids;   //! Array of current bids corresponding to bidders, index corresponds to targets @see _bidders
  IT*     _bidders; //! Array of current bidders corresponding to bids, index corresponds to targets @see _bids

  IT     _n;   //!< Size of $G_1$ and $G_2$


protected: /* PURE VIRTUAL MEMBER FUNCTIONS */

  /**
   * @brief The initialization phase
   *
   *    Basically allocate the arrays and give initial values to members.
   *
   * @note All parameters can be seen as initial values to begin with
   */
  virtual void initialization(const DT* C, const int& n, DT* u, DT* v, IT* G1_to_G2)
  {
    if (_unassigned) delete [] _unassigned;
    if (_bids)       delete [] _bids;
    if (_bidders)    delete [] _bidders;
    _nbUnassigned = n;
    _n = n;

    _unassigned = new bool[n];
    _bids = new double[n];
    _bidders = new IT[n];

    for (IT i=0; i<n; i++) _unassigned[i] = true;
    for (IT i=0; i<n; i++) _bids[i] = -1;
    for (IT i=0; i<n; i++) _bidders[i] = 0;
    for (IT i=0; i<n; i++) G1_to_G2[i] = -1;
    for (IT i=0; i<n; i++) u[i] = 0;
    for (IT i=0; i<n; i++) v[i] = 0;
  }


  /**
   * @brief The method to choose the set of bidders at each iteration
   *
   *    This can be one of the three possible approaches :
   *    * The <em>Gauss-Seidel</em> approach, where only one bidder is selected at the time
   *    * The <em>Jacobi</em> approach, where all vertex in $G_1$ are selected as bidders
   *    * A combination of both, with a number of bidder between 1 and $n$
   *
   * @return A list of vertex chosen as bidders
   */
  virtual std::list<IT> chooseBidders() const = 0;


  /**
   * @brief Implements the bidding round of the auction algorithm
   *
   *    This step updates the members <code>_bids</code> and <code>_bidder</code>
   *    regarding the behavior of each bidder in <code>bidders</code>
   *
   * @param C [in]   The cost matrix
   * @param v [in]  The dual variable associated to $G_2$, $i.e.$ the price of each vertex
   * @param bidders [in]     The list of the current bidders
   */
  virtual void biddingRound( const DT* C, DT* v, const std::list<IT>& bidders )
  {
    // find the best candidate ji for each i
    IT ji;   double pi;   //< best profit
    double ps;   //< second best profit
    for(typename std::list<IT>::const_iterator it=bidders.begin(); it != bidders.end(); it++){
      IT i = *it;
      pi = C[sub2ind(i,0,_n)] - v[0];
      ji = 0;
      // Best profit search
      for (IT j=1; j<_n; j++){
        if (C[sub2ind(i,j,_n)] - v[j] > pi){
          pi = C[sub2ind(i,j,_n)] - v[j];
          ji = j;
        }
      }
      // Second best profit search
      if (ji == 0 && _n > 1)
        ps = C[sub2ind(i,1,_n)] - v[1];
      else
        ps = C[sub2ind(i,0,_n)] - v[0];
      for (IT j=0; j<_n; j++){
        if (j != ji && C[sub2ind(i,j,_n)] - v[j] > ps){
          ps = C[sub2ind(i,j,_n)] - v[j];
        }
      }
      bid(*it, ji, pi - ps + _epsilon);

    }
  }


  /**
   * @brief Implements the matching round of the auction algorithm
   * @param C [in]    The cost matrix
   * @param u [inout] Benefits
   * @param v [inout] Prices
   * @param G1_to_G2  [inout]  Matching
   */
  virtual void matchingRound( const DT* C, DT* u, DT* v, IT* G1_to_G2 )
  {
    // for all bid > 0 :
    for (int j=0; j<_n; j++){
      if (_bids[j] > 0){
        // unmatch j if matched
        for (int i=0; i<_n; i++){
          if (G1_to_G2[i] == j) {
            _unassigned[i] = true;
            G1_to_G2[i] = -1;
            _nbUnassigned++;
            continue;
          }
        }
        // match the best bidder to j
        G1_to_G2[_bidders[j]] = j;
        _unassigned[_bidders[j]] = false;
        _nbUnassigned--;

        // update the dual variables
        v[j] += _bids[j];
        u[_bidders[j]] = C[sub2ind(_bidders[j],j,_n)] - v[j];
      }
    }
  }



protected: /* PROTECTED MEMBER FUNCTIONS */

  /**
   * @brief The given bidder tries to bid on <code>target</code> the bid <code>increment</code>
   *
   *    The procedure check if the given bid (<code>increment</code>) is greater than the
   *    current bid on target (if any). That way, only the biggest bid and the associate
   *    bidder is saved for each vertex in $G_2$.
   *
   * @param bidder
   * @param target
   * @param increment
   */
  virtual void bid(IT bidder, IT target, DT increment){
    if (_bids[target] < increment){
      _bids[target] = increment;
      _bidders[target] = bidder;
    }
  }


  /**
   * @brief The main auction algorithm, which calls to the different pure virtual steps
   * @param C   The cost matrix of size $n \times n$
   * @param n   Size of the data
   * @param u   Dual variables associated to $G_1$
   * @param v   Dual variables associated to $G_2$
   * @param G1_to_G2    The assignment
   */
  virtual void auctionAlgorithm( const DT* C, const int& n, DT* u, DT* v, IT* G1_to_G2 ){
    do{
      // Reset all bids
      for (int j=0; j<n; j++) _bids[j] = -1;

      std::list<IT> bidders = chooseBidders();
      biddingRound(C, v, bidders);
      matchingRound(C, u, v, G1_to_G2);
    } while (_nbUnassigned > 0);
  }


public: /* PUBLIC MEMBER FUNCTIONS */

  AuctionLSAP( const double& firstEpsilon,
               const double& finalEpsilon,
               const double& scalingFactor ):
    _epsilon(firstEpsilon), _finalEpsilon(finalEpsilon), _scalingFactor(scalingFactor),
    _unassigned(NULL), _nbUnassigned(0),
    _bids(NULL), _bidders(NULL)
  {}


  /**
   * finalEpsilon will be lower than $\frac{1}{n}$
   */
  AuctionLSAP( const double& firstEpsilon,
               const int& n ):
    _epsilon(firstEpsilon), _finalEpsilon(1.0/(n+1)), _scalingFactor(5.0),
    _unassigned(NULL), _nbUnassigned(0),
    _bids(NULL), _bidders(NULL)
  {}


  /**
   * Initialization as in [1] :
   * * firstEpsilon = max(C) / 5
   * * finalEpsilon = 1/(n+1)
   * * scaling factor = 5
   * [1] Castanon, Reverse auction algorithms for assignment problems, 1993
   */
  AuctionLSAP( const DT* C,
               const int& n ):
    _epsilon(0.0), _finalEpsilon(1.0/(n+1)), _scalingFactor(5.0),
    _unassigned(NULL), _nbUnassigned(0),
    _bids(NULL), _bidders(NULL)
  {
    DT max = C[0];
    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        if (C[sub2ind(i,j,n)] > max) max = C[sub2ind(i,j,n)];
      }
    }
    _epsilon = max / 5;
  }

  ~AuctionLSAP(){
    if (_unassigned) delete [] _unassigned;
    if (_bids)       delete [] _bids;
    if (_bidders)    delete [] _bidders;
  }


  /**
   * @brief Apply the sequential auction algorithm several times with epsilon scaling.
   *
   *    Returns the dual variables $u$ and $v$ as well as and assignment in G1_to_G2
   *
   * @param C   The cost matrix of size $n \times n$
   * @param n   Size of the data
   * @param u   Dual variables associated to $G_1$
   * @param v   Dual variables associated to $G_2$
   * @param G1_to_G2    The assignment
   */
  void operator() ( const DT* C, const int& n, DT* u, DT* v, IT* G1_to_G2 ){
  initialization(C, n, u, v, G1_to_G2);
    _epsilon *= _scalingFactor;
    do{
      _epsilon = _epsilon / _scalingFactor;

      for (int i=0; i<n; i++){
        G1_to_G2[i] = -1;
        _unassigned[i] = true;
        _nbUnassigned = n;
      }

      auctionAlgorithm(C, n, u, v, G1_to_G2);
    }while(_epsilon > _finalEpsilon);
  }


};


#endif
