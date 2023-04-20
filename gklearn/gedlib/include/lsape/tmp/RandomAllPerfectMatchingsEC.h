/**
 * @file AllPerfectMatchings.h
 * @author Évariste <<21511575@etu.unicaen.fr>>
 */

#ifndef __RANDOMALLPERFECTMATCHINGS_H__
#define __RANDOMALLPERFECTMATCHINGS_H__

#include <random>
#include <chrono>
#include "AllPerfectMatchingsEC.h"

/**
 * @class RandomAllPerfectMatchingsEC
 * @autor evariste
 * @brief Error correcting version of AllPerfectMatchings with randomize order of enumeration
 *
 *   At each step of Uno's algorithm, the alternate cycle is randomly chosen
 */
template <typename IT>
class RandomAllPerfectMatchingsEC : public AllPerfectMatchingsEC<IT>
{

private:
  
  std::default_random_engine randGen;

protected:
  
  virtual bool
  chooseEdge( const cDigraph<IT>& gm, IT* x, IT* y );
  
  
public :

  RandomAllPerfectMatchingsEC(cDigraph<IT>& gm, const IT& n, const IT& m) : 
    AllPerfectMatchingsEC<IT>(gm, n, m)
  {
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned int seed = 221645753;
    randGen.seed(seed);
  }
  
};


/*********** Implementation ***************/

/*
template <typename IT>
bool
RandomAllPerfectMatchingsEC<IT>::chooseEdge( const cDigraph<IT>& gm, IT* x, IT* y ) 
{
  int idx;
  
  while(this->scc.size() > 0){
    std::uniform_int_distribution<int> distribution(0,this->scc.size()-1);
    idx = distribution(randGen); 
    typename std::list<BipartiteSCC>::iterator it = (this->scc).begin();
    for (IT i=0; i<idx; i++) it++;
    
    //IT i=0; while(i<it->u.size() && it->u[i] != true) i++;
    //IT j=0; while(j<it->v.size() && it->v[j] != true) j++;
    
    for (IT i=0; i<gm.rows(); i++){
      if (it->u[i]){
        for (IT j=0; j<gm.cols(); j++){
          if (it->v[j] && (i<this->_n || j<this->_m) && gm(i,j) == 1){
            *x = i;
            *y = j;
            return true;
          }
        }
      }
    }
    
    // if we reach this point, this scc is integrally in eps-to-eps quarter
    this->scc.erase(it);
  }
  
  // there is no scc outside eps-to-eps submatrix
  return false;
}
//*/

template <typename IT>
bool
RandomAllPerfectMatchingsEC<IT>::chooseEdge( const cDigraph<IT>& gm, IT* x, IT* y ) 
{
  int idx;
  
  while(this->scc.size() > 0){
    std::uniform_int_distribution<int> distribution(0,this->scc.size()-1);
    idx = distribution(randGen); 
    typename std::list<BipartiteSCC>::iterator it = (this->scc).begin();
    for (IT i=0; i<idx; i++) it++;
    
    IT i=0; IT j=0; bool erase=false;
    while(i<this->_n && it->u[i] != true) i++;
    if (i == this->_n){
      while(j<this->_m && it->v[j] != true) j++;
      if (j == this->_m){
        erase = true;
      }
    }
    
    if (!erase){
      // how many lines in the scc ?
      int nl=0; for(int i=0; i<it->u.size(); i++) nl += (int)(it->u[i]);
      std::uniform_int_distribution<int> distrib(0,nl);
      int idx_line = distrib(randGen);
      
      // find the idx_line '1'
      nl=0;
      for(int i=0; i<it->u.size(); i++){
        nl += (int)(it->u[i]);
        if (nl == idx_line){
          *x = i;
          for (int j=0; j<this->_m; j++){
            if (gm(i,j) == 1){
              *y = j;
              return true;
            }
          }
        }
      }
    }
    
    this->scc.erase(it);
    
  }
  
  // there is no scc outside eps-to-eps submatrix
  return false;
}

#endif

