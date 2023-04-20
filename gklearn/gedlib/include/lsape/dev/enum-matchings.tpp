/**
 * @file AllPerfectMatchings.tpp
 * @author Évariste <<21511575@etu.unicaen.fr>>
 *
 * @todo
 */

#include <iostream>

namespace lsape {

template <typename IT>
void AllPerfectMatchings<IT>::strongConnect( const cDigraph<IT>& gm, int v )
{
  vnum[v+offset] = num;
  vaccess[v+offset] = num;
  num += 1;

  tarjanStack.push(v);
  setstack.push(!(bool)(offset));
  instack[v+offset] = true;

  // For each w successor of v
  // 2 cases :
  // * v in X => 1 successor,  gm[v,w] ==  1
  // * v in Y => * successors, gm[w,v] == -1

  int w=0;
  if (offset == 0){ // v in X : find the 1
    while(w < (int)(gm.cols())){
      if (gm(v,w) == 1){
        if (vnum[w+offsize] == -1){ // if num(w) not defined
          // explore w
          offset = offsize; // w is in Y
          strongConnect(gm, w);
          offset = 0; // we are back in X
          vaccess[v] = std::min(vaccess[v], vaccess[w+offsize]);
        }
        else if(instack[w+offsize]){
          vaccess[v] = std::min(vaccess[v], vnum[w+offsize]);
        }
        //bypass the rest of search (there is max one successor)
        w = gm.cols();
      }
      w++;
    }
  }
  else{ // v in Y : find all the -1
    while(w < (int)(gm.rows())){
      if (gm(w,v) == -1){ // if w is a successor
        if (vnum[w] == -1){
          offset = 0; // w is in X
          strongConnect(gm, w);
          offset = offsize; // back in Y
          vaccess[v+offset] = std::min(vaccess[v+offset], vaccess[w]);
        }
        else if(instack[w]){
          vaccess[v+offset] = std::min(vaccess[v+offset], vnum[w]);
        }
      }
      w++;
    }
  }

  // If v is a root (v.access == v.num)
  if (vaccess[v+offset] == vnum[v+offset]){
    //scc.emplace_back(gm.rows(), gm.cols());
    BipartiteSCC _bpscc;
    _bpscc.u.resize(gm.rows(), false);
    _bpscc.v.resize(gm.cols(), false);
    scc.push_back(_bpscc);

    bool inX = setstack.top();
    do{
      if (!tarjanStack.empty()){
        w = tarjanStack.top();
        tarjanStack.pop();

        inX = setstack.top();
        setstack.pop();
        instack[w+offset] = false;

        if(inX)
          scc.back().u[w] = true;
        else
          scc.back().v[w] = true;
      }
    }while(w != v || !(bool)(offset) != inX);
  }
} // end strongConnect()


template <typename IT>
const std::list<BipartiteSCC>&
AllPerfectMatchings<IT>::findSCC( const cDigraph<IT>& gm ){

  // Initialization
  num = 0;
  access = 0;
  vnum.clear();
  vaccess.clear();
  instack.clear();

  scc.clear();

  vnum.resize(gm.rows()+gm.cols(), -1);
  instack.resize(gm.rows()+gm.cols(), false);
  vaccess.resize(gm.rows()+gm.cols());

  // Begin in X
  offset = 0;
  offsize = gm.rows();


  // For each vertice in X (rows of gm)
  for (IT i=0; i<gm.rows(); i++){
    if (vnum[i] == -1){
      offset = 0;
      strongConnect(gm, i);
    }
  }

  return scc;
}


/*XXX comparer avec la méthode de seb avec les listes :
 * std::list<BipartiteSCC> avec BipartiteSCC = {list<int>, list<int>}
 * contient les id des sommets dans les scc    (dans X   ,   dans Y)
 */
 template <typename IT>
std::list<_Edge<IT> >
AllPerfectMatchings<IT>::rmUnnecessaryEdges( cDigraph<IT>& gm, const std::list<BipartiteSCC>& scc_in ){
  std::list<_Edge<IT> > res;
  bool inSomeSCC;
  for (IT i=0; i<gm.rows(); i++){
    for (IT j=0; j<gm.cols(); j++){
      if (gm(i,j) != 0){
        inSomeSCC = false;
        for(std::list<BipartiteSCC>::const_iterator it=scc_in.begin(); it!=scc_in.end(); it++)
          inSomeSCC = inSomeSCC || (it->u[i] && it->v[j]);

        if (!inSomeSCC){
          gc(i,j) = gm(i,j);
          gm(i,j) = 0;
          res.emplace_back(i,j);
        }
      }
    }
  }
  return res;
}


template <typename IT>
std::list<_Edge<IT> >
AllPerfectMatchings<IT>::rmUnnecessaryEdges( cDigraph<IT>& gm){
  return rmUnnecessaryEdges(gm, scc);
}


template <typename IT>
void
AllPerfectMatchings<IT>::enumPerfectMatchings( cDigraph<IT>& gm, int k, int maxDp){
  gc.re_alloc(gm.rows(), gm.cols());
  maxmatch = k;
  nbmatch = 0; // if we are here, we know at least one matching
  depth = 0;
  maxdepth = maxDp;
  saveMatching(gm);
  enumPerfectMatchings_iter(gm);
}


template <typename IT>
void
AllPerfectMatchings<IT>::enumPerfectMatchings_iter( cDigraph<IT>& gm ){
  // Stop if enough matchings are found
  if (maxmatch != -1 && nbmatch >= maxmatch) return;
  if (maxdepth != -1 && depth >= maxdepth) return;

  // We get deeper in the research tree
  depth++;

  findSCC(gm);

  /* remove epsilon-only SCCs
  for(std::list<BipartiteSCC>::iterator it=scc.begin(); it!=scc.end(); it++){
    int i=0; while(i<it->u.size()/2 && !it->u[i] && !it->v[i]) i++;
    if (i == gm.rows()/2 ){
      it = scc.erase(it);
    }
  }
  //*/

  std::list<_Edge<IT> > delEdges = rmUnnecessaryEdges(gm);

  // execute only if there is an edge in the graph
  IT x, y;
  if (chooseEdge( gm, &x, &y )){

    // we consider a cycle in a scc containing (x,y)
    std::list< _Edge<IT> > cycle;
    cycle.emplace_back(x,y);


    std::vector<bool> visited(gm.rows(), false);
    depthFirstSearchCycle(gm, x, cycle, visited);
    cycle.pop_back();

    // Get the new matching m' and apply iter on Gm'-(e)
    reverseEdges(gm, cycle);
    saveMatching(gm + gc);

    // Save the edge that will be erased to construct G- in place in gc
    gc(x,y) = gm(x,y);
    gm(x,y) = 0;

    enumPerfectMatchings_iter(gm);

    // Reconstruct the edge and refind m from m'
    gm(x,y) = gc(x,y);
    gc(x,y) = 0;
    reverseEdges(gm, cycle);

    // Save the vectors that will be erased to construct G+ in gc and locally
    std::vector<IT> ux(gm.cols());
    std::vector<IT> uy(gm.rows());
    IT i; int gmxy = gm(x,y);
    for (i=0; i<gm.cols(); i++){
      gc(x,i) += gm(x,i);
      ux[i] = gm(x,i);
      gm(x,i) = 0;
    }
    for (i=0; i<gm.rows(); i++){
      gc(i,y) += gm(i,y);
      uy[i] = gm(i,y);
      gm(i,y) = 0;
    }
    gc(x,y) = gmxy;

    enumPerfectMatchings_iter(gm);


    // reconstruct the graph and reset the corresponding values in gc
    for (i=0; i<gm.cols(); i++){
      gm(x,i) = ux[i];
      gc(x,i) -= ux[i];
    }
    for (i=0; i<gm.rows(); i++){
      gm(i,y) = uy[i];
      gc(i,y) -= uy[i];
    }
    gm(x,y) = gmxy;
    gc(x,y) = 0;
  } // endif chooseEdge

  // Reconstruct unnecessary edges

  for(typename std::list< _Edge<IT> >::iterator it=delEdges.begin();
      it != delEdges.end(); it++){
    gm(it->x, it->y) = gc(it->x, it->y);
    gc(it->x, it->y) = 0;
  }

  depth--;

}



/************ equality digraph *************/
template <typename DT, typename IT>
cDigraph<IT> equalityDigraph(const DT *C, const IT &n, const IT &m,
                                            const IT *rho12, const DT *u, const DT *v){

  Digraph<char,IT> edg(n,m);

  for (IT j=0; j<m; j++){
    for (IT i=0; i<n; i++){
      if (rho12[i] == j) // a matching edge
        edg(i,j) = 1;
      else{
        const DT &c = C[j*n+i];
        if ( c<0 ) edg(i,j) = 0; // forbidden matching edge
        if (c - (u[i] + v[j]) == 0)
          edg(i,j) = -1;
        else edg(i,j) = 0;
      }
    }
  }
  return edg;
}


template <typename DT, typename IT>
cDigraph<IT> greedySortDigraph(const DT *C, const IT &nrows, const IT &ncols){
  
  int n = nrows-1;
  int m = ncols-1;
  Digraph<char,IT> dg(n+m,m+n);

  
  DT minc = C[0], maxc = C[0];
  const DT *ite = C+nrows*ncols-1;
  for (const DT *itc = C+1; itc < ite; itc++)
  {
    const DT &vc = *itc;
    if (vc < minc) minc = vc;
    else if (vc > maxc) maxc = vc;
  }

  // construct histogram
  IT nbins = maxc - minc + 1;
  IT *bins = new IT[nbins+1];
  const IT *itbe = bins+nbins;
  for (IT *itb = bins; itb < itbe; itb++) *itb = 0;
  for (const DT *itc = C; itc < ite; itc++) bins[*itc-minc]++;

  // starting index for each cost value
  IT tot = 0, oldcount;
  for (IT i = 0; i < nbins; i++) { oldcount = bins[i]; bins[i] = tot; tot += oldcount; }

  // reoder the costs, preserving order of C with equal keys and favor substitutions
  IT *idxs = new IT[nrows*ncols], k = 0;
  for (const DT *itc = C; itc < ite; itc++, k++) { idxs[bins[*itc-minc]] = k; bins[*itc-minc]++; }

  // assign element in ascending order of the costs
  IT *pt_idxs_end = idxs+nrows*ncols-1, i = 0, j = 0, approx = 0;
  IT *rho = new IT[nrows]; IT *varrho = new IT[ncols];
  for (; i < nrows-1; i++) rho[i] = -1;
  for (; j < ncols-1; j++) varrho[j] = -1;
  
  IT *pt_idx = idxs;
  for (IT nbc = 0; pt_idx != pt_idxs_end && nbc < nrows+ncols-2; pt_idx++)
  {
    ind2sub(*pt_idx,nrows,i,j);
    if (i == nrows-1) { if (varrho[j] == -1) { varrho[j] = i; approx += C[*pt_idx]; nbc++; dg(j+n,j) = 1; } else dg(j+n,j)=-1; }
    else
      if (rho[i] == -1)
        if (j == ncols-1) { rho[i] = j; approx += C[*pt_idx]; nbc++; dg(i,i+m) = 1;}
        else
        {
          if (varrho[j] == -1) { rho[i] = j; varrho[j] = i; approx += C[*pt_idx]; nbc += 2; dg(i,j) = 1;}
          else dg(i,j) = -1;
        }
      else
        if (j == ncols-1) dg(i,i+m)=-1;
        else dg(i,j) = -1;
  }
  
  DT cstop = C[*(pt_idx-1)];
  for (; pt_idx != pt_idxs_end && C[*pt_idx] == cstop; pt_idx++) {
    ind2sub(*pt_idx,nrows,i,j);
    if (i<n && j<m)       dg(i,j) = -1;
    else if(i==n && j<m)  dg(j+n,j) =-1;
    else if(i<n && j==m)  dg(i,i+m) =-1;
  }


  // Complete the matching with eps->eps edges
  if (m >= n){
      j = 0;
      for(IT i=0; i<m; i++){
        if (dg(i+n,i) != 1){
          while (j<n && dg(j,j+m) == 1) j++;
          dg(i+n,j+m) = 1;
          j++;
        }
      }
  }
  else{
      i = 0;
      for(IT j=0; j<n; j++){
        if (dg(j,j+m) != 1){
          while (i<m && dg(i+n,i) == 1) i++;
          dg(i+n,j+m) = 1;
          i++;
        }
      }
  }
  
  delete[] idxs; delete[] bins;
  delete[] rho; delete[] varrho;
  
  return dg;
}

template <typename IT>
cDigraph<IT> greedySortDigraph(const double *C, const IT &nrows, const IT &ncols) {}

template <typename IT>
cDigraph<IT> greedySortDigraph(const float *C, const IT &nrows, const IT &ncols) {}


/************ misc - accessors - constructors ************/

/*
BipartiteSCC::BipartiteSCC(unsigned int size_u, unsigned int size_v):
  u(size_u, false),
  v(size_v, false)
{}
//*/

template <typename IT>
_Edge<IT>::_Edge(IT _x, IT _y):
  x(_x),
  y(_y)
{}


template <typename IT>
const std::list< IT* >&
AllPerfectMatchings<IT>::getPerfectMatchings() const{
  return perfectMatchings;
}


template <typename IT>
bool
AllPerfectMatchings<IT>::chooseEdge( const cDigraph<IT>& gm, IT* x, IT* y ) {
  for (IT i=0; i<gm.rows(); i++){
    for (IT j=0; j<gm.cols(); j++){
      if (gm(i,j) == 1){
        *x = i;
        *y = j;
        return true;
      }
    }
  }
  return false;
}


template <typename IT>
void
AllPerfectMatchings<IT>::reverseEdges( cDigraph<IT>& gm, const std::list< _Edge<IT> >& edges) const{
  for(typename std::list<_Edge<IT> >::const_iterator it=edges.begin(); it != edges.end(); it++){
    gm(it->x, it->y) *= -1;
  }
}


template <typename IT>
void
AllPerfectMatchings<IT>::saveMatching(const cDigraph<IT>& gm ){
  IT* newmatch = new IT[gm.rows()];
  perfectMatchings.push_back(newmatch);
  IT j;
  for(IT i=0; i<gm.rows(); i++){
    j=0;
    while (j<gm.cols() && gm(i,j) != 1) j++;
    perfectMatchings.back()[i] = j;
  }
  nbmatch++;
}

template <typename IT>
bool
AllPerfectMatchings<IT>::depthFirstSearchCycle(
        const cDigraph<IT>& gm,
        const IT& x, std::list< _Edge<IT> >& cycle,
        std::vector<bool>& visited) const{

  IT v = cycle.back().y;

  IT i;
  for (i=0; i<gm.rows(); i++){
    if (!visited[i] && gm(i,v) == -1){
      visited[i] = true;

      cycle.emplace_back(i,v);
      IT j=0; while(j<gm.cols() && gm(i,j) != 1) j++;
      cycle.emplace_back(i,j);

      if(i == x || depthFirstSearchCycle(gm,x,cycle,visited)) return true;
      else{
        cycle.pop_back();
        cycle.pop_back();
      }
    }
  }
  if (i == gm.rows()) return false; // Dead end
  return false;
}


template <typename IT>
AllPerfectMatchings<IT>::AllPerfectMatchings(cDigraph<IT>& gm):
  vnum(gm.rows()+gm.cols(), -1),
  vaccess(gm.rows()+gm.cols()),
  instack(gm.rows()+gm.cols(), false),
  perfectMatchings(_perfectMatchings),
  _pMatchInClass(true),
  offset(0),
  offsize(gm.rows()),
  gc(gm.rows(), gm.cols()),
  nbmatch(0)
{}

template <typename IT>
AllPerfectMatchings<IT>::AllPerfectMatchings(cDigraph<IT>& gm, std::list<IT*> &pMatchings):
  vnum(gm.rows()+gm.cols(), -1),
  vaccess(gm.rows()+gm.cols()),
  instack(gm.rows()+gm.cols(), false),
  perfectMatchings(pMatchings),
  _pMatchInClass(false),
  offset(0),
  offsize(gm.rows()),
  gc(gm.rows(), gm.cols()),
  nbmatch(0)
{ }

template <typename IT>
AllPerfectMatchings<IT>::~AllPerfectMatchings(){
  // free the allocated memory of perfectMatchings
  if (_pMatchInClass) for(typename std::list<IT*>::const_iterator it=perfectMatchings.begin(); it != perfectMatchings.end(); it++) if (*it) delete[] *it;
}


template<typename IT>
void AllPerfectMatchings<IT>::deleteMatching(){
  for(typename std::list<IT*>::iterator it=perfectMatchings.begin(); it != perfectMatchings.end(); it++) { delete[] *it; *it = NULL; }
  perfectMatchings.clear();
}

} // namespace lsape
