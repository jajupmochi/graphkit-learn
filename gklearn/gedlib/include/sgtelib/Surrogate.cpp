/*-------------------------------------------------------------------------------------*/
/*  sgtelib - A surrogate model library for derivative-free optimization               */
/*  Version 2.0.1                                                                      */
/*                                                                                     */
/*  Copyright (C) 2012-2017  Sebastien Le Digabel - Ecole Polytechnique, Montreal      */ 
/*                           Bastien Talgorn - McGill University, Montreal             */
/*                                                                                     */
/*  Author: Bastien Talgorn                                                            */
/*  email: bastientalgorn@fastmail.com                                                 */
/*                                                                                     */
/*  This program is free software: you can redistribute it and/or modify it under the  */
/*  terms of the GNU Lesser General Public License as published by the Free Software   */
/*  Foundation, either version 3 of the License, or (at your option) any later         */
/*  version.                                                                           */
/*                                                                                     */
/*  This program is distributed in the hope that it will be useful, but WITHOUT ANY    */
/*  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    */
/*  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   */
/*                                                                                     */
/*  You should have received a copy of the GNU Lesser General Public License along     */
/*  with this program. If not, see <http://www.gnu.org/licenses/>.                     */
/*                                                                                     */
/*  You can find information on sgtelib at https://github.com/bastientalgorn/sgtelib   */
/*-------------------------------------------------------------------------------------*/

#include "Surrogate.hpp"

using namespace SGTELIB;

/*--------------------------------------*/
/*              constructor             */
/*--------------------------------------*/

SGTELIB::Surrogate::Surrogate ( SGTELIB::TrainingSet & trainingset,
                                const SGTELIB::Surrogate_Parameters param) :
  _trainingset ( trainingset      ) ,
  _param       ( param            ) ,   
  _n           (_trainingset.get_input_dim()  ) ,
  _m           (_trainingset.get_output_dim() ) ,
  _p_ts        (0                 ) ,
  _p_ts_old    (999999999         ) ,
  _p           (0                 ) ,
  _p_old       (999999999         ) ,
  _ready       (false             ) ,
  _Zhs         (NULL              ) ,
  _Shs         (NULL              ) ,
  _Zvs         (NULL              ) ,
  _Svs         (NULL              ) ,
  _selected_points (1,-1          ) ,
  _metric_emax     (NULL          ) ,
  _metric_emaxcv   (NULL          ) ,
  _metric_rmse     (NULL          ) ,
  _metric_rmsecv   (NULL          ) ,
  _metric_oe       (NULL          ) ,
  _metric_oecv     (NULL          ) ,
  _metric_linv     (NULL          ) ,
  _metric_aoe      (-1.0          ) ,
  _metric_aoecv    (-1.0          ) ,
  _metric_armse    (-1.0          ) ,
  _metric_armsecv  (-1.0          ) ,
  _psize_max       ( 0.5          ) ,
  _out             (              ) ,
  _display         ( false        ) {;
}//


SGTELIB::Surrogate::Surrogate ( SGTELIB::TrainingSet & trainingset,
                                const SGTELIB::model_t mt ) :
  _trainingset ( trainingset      ) ,
  _param       ( mt ) ,   
  _n     (_trainingset.get_input_dim()  ) ,
  _m     (_trainingset.get_output_dim() ) ,
  _p_ts      (0                   ) ,
  _p_ts_old  (999999999           ) ,
  _p         (0                   ) ,
  _p_old     (999999999           ) ,
  _ready (false                   ) ,
  _Zhs   (NULL                    ) ,
  _Shs   (NULL                    ) ,
  _Zvs   (NULL                    ) ,
  _Svs   (NULL                    ) ,
  _selected_points (1,-1          ) ,
  _metric_emax     (NULL          ) ,
  _metric_emaxcv   (NULL          ) ,
  _metric_rmse     (NULL          ) ,
  _metric_rmsecv   (NULL          ) ,
  _metric_oe       (NULL          ) ,
  _metric_oecv     (NULL          ) ,
  _metric_linv     (NULL          ) ,
  _metric_aoe      (-1.0          ) ,
  _metric_aoecv    (-1.0          ) ,
  _metric_efioe    (-1.0          ) ,
  _metric_efioecv  (-1.0          ) ,
  _metric_armse    (-1.0          ) ,
  _metric_armsecv  (-1.0          ) ,
  _psize_max       ( 0.5          ) ,
  _out             (              ) ,
  _display         ( false        ) {
}//

SGTELIB::Surrogate::Surrogate ( SGTELIB::TrainingSet & trainingset,
                                const std::string & s) :
  _trainingset ( trainingset      ) ,
  _param       ( s                ) ,   
  _n     (_trainingset.get_input_dim()  ) ,
  _m     (_trainingset.get_output_dim() ) ,
  _p_ts      (0                   ) ,
  _p_ts_old  (0                   ) ,
  _p         (0                   ) ,
  _p_old     (0                   ) ,
  _ready (false                   ) ,
  _Zhs   (NULL                    ) ,
  _Shs   (NULL                    ) ,
  _Zvs   (NULL                    ) ,
  _Svs   (NULL                    ) ,
  _selected_points (1,-1          ) ,
  _metric_emax     (NULL          ) ,
  _metric_emaxcv   (NULL          ) ,
  _metric_rmse     (NULL          ) ,
  _metric_rmsecv   (NULL          ) ,
  _metric_oe       (NULL          ) ,
  _metric_oecv     (NULL          ) ,
  _metric_linv     (NULL          ) ,
  _metric_aoe      (-1.0          ) ,
  _metric_aoecv    (-1.0          ) ,
  _metric_efioe    (-1.0          ) ,
  _metric_efioecv  (-1.0          ) ,
  _metric_armse    (-1.0          ) ,
  _metric_armsecv  (-1.0          ) ,
  _psize_max       ( 0.5          ) ,
  _out             (              ) ,
  _display         ( false        ) {
}//


/*--------------------------------------*/
/*               destructor             */
/*--------------------------------------*/
SGTELIB::Surrogate::~Surrogate ( void ) {
  reset_metrics();
}//


void SGTELIB::Surrogate::info ( void ) const {
  _trainingset.info();
}//


/*--------------------------------------*/
/*              display                 */
/*--------------------------------------*/
void SGTELIB::Surrogate::display ( std::ostream & out ) const {
  out << "Surrogate: " << get_string() << "\n";
  out << "ready: " << _ready << "\n";
  out << "n: " << _n << " (input dim)\n";
  out << "m: " << _m << " (output dim)\n";
  out << "p: " << _p << " (nb points)\n";
  display_private ( out );
}//

/*--------------------------------------*/
/*       erase_data                     */
/*--------------------------------------*/
void SGTELIB::Surrogate::reset_metrics ( void ) {
  #ifdef SGTELIB_DEBUG
    std::cout << "Surrogate: reset_metrics...";
  #endif

  if (_Zhs) delete _Zhs;
  _Zhs = NULL;  

  if (_Shs) delete _Shs;
  _Shs = NULL;  

  if (_Zvs) delete _Zvs;
  _Zvs = NULL;  

  if (_Svs) delete _Svs;
  _Svs = NULL;  

  if (_metric_emax)   delete [] _metric_emax;
  _metric_emax = NULL;

  if (_metric_emaxcv) delete [] _metric_emaxcv;
  _metric_emaxcv = NULL;

  if (_metric_rmse)   delete [] _metric_rmse;
  _metric_rmse = NULL;

  if (_metric_rmsecv) delete [] _metric_rmsecv;
  _metric_rmsecv = NULL;

  if (_metric_oe)     delete [] _metric_oe;
  _metric_oe = NULL;

  if (_metric_oecv)   delete [] _metric_oecv;
  _metric_oecv = NULL;

  if (_metric_linv)   delete [] _metric_linv;
  _metric_linv = NULL;

  _metric_aoe     = -1.0;
  _metric_aoecv   = -1.0;
  _metric_efioe   = -1.0;
  _metric_efioecv = -1.0;
  _metric_armse   = -1.0;
  _metric_armsecv = -1.0;

  #ifdef SGTELIB_DEBUG
    std::cout << "OK\n";
  #endif
}//

/*--------------------------------------*/
/*               build                  */
/*--------------------------------------*/
bool SGTELIB::Surrogate::build ( void ) {

  #ifdef SGTELIB_DEBUG
    std::cout << "Surrogate build - BEGIN\n";
  #endif

  if (streqi(_param.get_output(),"NULL")){
    _display = false;
  }
  else{
    _display = true;
  } 

  // Check the parameters of the model:
  _param.check();

  // Before building the surrogate, the trainingset must be ready
  _trainingset.build();

  // Number of points in the training set.
  _p_ts = _trainingset.get_nb_points();
  //std::cout << _ready << " " << _p_ts << " " << _p_ts_old << "\n";
  if ( (_ready) && (_p_ts==_p_ts_old) ){
    #ifdef SGTELIB_DEBUG
      std::cout << "Surrogate build - SKIP Build\n";
    #endif
    return true;
  }
  
  // Otherwise, the model is not ready and we need to call build_private
  _ready = false;


  // Get the number of points used in the surrogate
  if ( (_selected_points.size()==1) && (_selected_points.front()==-1) )
    _p = _p_ts;
  else  
    _p = static_cast<int>(_selected_points.size());

  // Need at least 2 point to build a surrogate.
  if (_p<2){
    return false;
  }

  // Delete the intermediate data and metrics 
  // (they will have to be recomputed...)
  reset_metrics();

  // If there are new points, 
  // Call to the private build
  #ifdef SGTELIB_DEBUG
    std::cout << "Surrogate build - BUILD_PRIVATE\n";
  #endif

  bool ok;
  ok = init_private();
  if ( ! ok) return false;

  // Optimize parameters
  if (_param.get_nb_parameter_optimization()>0){
    ok = optimize_parameters();
    if ( ! ok){
      _ready = false;
      return false;
    }
  }

  // Build private
  ok = build_private();
  if ( ! ok){
    _ready = false;
    return false;
  }


  // Memorize previous number of points
  _p_ts_old = _p_ts;
  _p_old = _p;

  #ifdef SGTELIB_DEBUG
    std::cout << "Surrogate build - END\n";
  #endif
  
  if (_display){
    _out.open(_param.get_output().c_str() , std::ios::out | std::ios::app);
    if (_out.fail()) std::cout << "Out.fail1!!!\n";
    std::cout << "Write in " << _param.get_output() << "\n";
    if (_out.fail()) std::cout << "Out.fail2!!!\n";
    display(_out);
    if (_out.fail()) std::cout << "Out.fail3!!!\n";
    //_out << "AOECV: " << get_metric(SGTELIB::METRIC_AOECV,0) << "\n";
    //_out << "ARMSECV: " << get_metric(SGTELIB::METRIC_ARMSECV,0) << "\n";
    _out.close();
  }


  _ready = true;
  return true;
}//

bool SGTELIB::Surrogate::init_private (void) {
  // Empty initialization function
  #ifdef SGTELIB_DEBUG
    std::cout << model_type_to_str(get_type()) << " : init_private\n";
  #endif
  return true;
}


/*--------------------------------------*/
/*               check_ready            */
/*--------------------------------------*/
void SGTELIB::Surrogate::check_ready (void) const {
    check_ready("");
}//

/*--------------------------------------*/
void SGTELIB::Surrogate::check_ready (const std::string & file,
                                      const std::string & function,
                                      const int & i        ) const {
    check_ready(file+"::"+function+"::"+itos(i));
}//
/*--------------------------------------*/
void SGTELIB::Surrogate::check_ready (const std::string & s) const {
  
  // Check the tag _ready
  if ( ! _ready){
    display(std::cout);
    std::cout << "Surrogate: NOT READY! (" << s << ")\n";
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                 "check_ready(): Not ready!" );
  }

  // Check if the trainingset is ready
  _trainingset.check_ready("From Surrogate ()");


  // Check the new number of points in the trainingset
  if (_trainingset.get_nb_points()>_p_ts){
    display(std::cout);
    std::cout << "Surrogate: NOT READY! (" << s << ")\n";
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                 "check_ready(): Not ready!" );
  }

}//


/*--------------------------------------*/
/*               add points             */
/*--------------------------------------*/
bool SGTELIB::Surrogate::add_points ( const SGTELIB::Matrix & Xnew ,
                                      const SGTELIB::Matrix & Znew  ){
  throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
       "add_points: forbiden." );
  return _trainingset.add_points(Xnew,Znew);
}//
/*--------------------------------------*/
bool SGTELIB::Surrogate::add_point  ( const double * xnew ,
                                      const double * znew  ){
  throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
       "add_point: forbiden." );
  return _trainingset.add_point(xnew,znew);
}//


/*--------------------------------------*/
/*               predict                */
/*--------------------------------------*/
void SGTELIB::Surrogate::predict ( const SGTELIB::Matrix & XX ,
                                         SGTELIB::Matrix * ZZ ,
                                         SGTELIB::Matrix * std, 
                                         SGTELIB::Matrix * ei ,
                                         SGTELIB::Matrix * cdf) {

  check_ready(__FILE__,__FUNCTION__,__LINE__);

  //std::cout << "IN PREDICT (public) " << __FILE__ << " " <<  ZZ << " " << std << " " << ei << " " << cdf << "\n";

  // Check the number of columns in XX
  if (XX.get_nb_cols() != _n){
    display(std::cout);
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                 "predict(): dimension error" );
  }

  *ZZ = SGTELIB::Matrix("ZZ",XX.get_nb_rows(),_m);

  // Scale the input
  SGTELIB::Matrix XXs(XX);
  XXs.set_name("XXs");
  _trainingset.X_scale(XXs);

  if (ei){
    ei->fill(-INF);
  }

  // Call the private prediction with normalize input XXs
  predict_private( XXs , ZZ , std , ei , cdf );

  // If nbdiff==1, put the values to 0.0
  int pxx = XX.get_nb_rows();
  if (ZZ){
    for (int j=0 ; j<_m ; j++){
      if (_trainingset.get_Z_nbdiff(j)==1){
        for (int i=0 ; i<pxx ; i++){
          ZZ->set(i,j,0.0);
        }
      }
    }
  }


  #ifdef SGTELIB_DEBUG
    if (ZZ){
      if (ZZ->has_nan()){
        ZZ->replace_nan (+INF);
      }
    }
    if (std){
      if (std->has_nan()){
        display(std::cout); 
        throw SGTELIB::Exception ( __FILE__ , __LINE__ , "predict(): std has nan" );
      }
    }
    if (ei){
      if (ei->has_nan()){
        display(std::cout); 
        throw SGTELIB::Exception ( __FILE__ , __LINE__ , "predict(): ei has nan" );
      }
    }
    if (cdf){
      if (cdf->has_nan()){
        display(std::cout); 
        throw SGTELIB::Exception ( __FILE__ , __LINE__ , "predict(): cdf has nan" );
      }
    }
  #endif

  ZZ->replace_nan (+INF);
  std->replace_nan (+INF);
  ei->replace_nan (-INF);
  cdf->replace_nan (0);

  // UnScale the output
  if (ZZ ){
    ZZ->set_name("ZZ");   
    _trainingset.Z_unscale(ZZ);
  }
  if (std){
    std->set_name("std");
    _trainingset.ZE_unscale(std);
  }
  if (ei ){
    ei->set_name("ei");
    _trainingset.ZE_unscale(ei);
    // ei is only computed for the OBJ output, so the other values are dummy, 
    // So we put them all to 0.    
    for (int j=0 ; j<_m ; j++){ 
      if (_trainingset.get_bbo(j)!=SGTELIB::BBO_OBJ){
        for (int i=0 ; i<pxx ; i++){ 
          ei->set(i,j,0.0);
        }
      }
    }  
  }
  if (cdf){
    cdf->set_name("cdf");
  }


}//



/*--------------------------------------*/
/*       predict (ZZs,std,ei)           */
/*--------------------------------------*/
// This function is the default method to compute std, ei and cdf.
// It can be overloaded, but models PRS, RBF and KS use the default method.
// This method relies on the private method predict_private(XXs,ZZs)
// which HAS TO be overloaded (pure virtual)
void SGTELIB::Surrogate::predict_private (const SGTELIB::Matrix & XXs,
                                                SGTELIB::Matrix * ZZs,
                                                SGTELIB::Matrix * std, 
                                                SGTELIB::Matrix * ei ,
                                                SGTELIB::Matrix * cdf) {
  check_ready(__FILE__,__FUNCTION__,__LINE__);


  const int pxx = XXs.get_nb_rows();
  const double fs_min = _trainingset.get_fs_min();
  int i,j;

  // Prediction of ZZs
  if ( (ZZs) || (ei) || (cdf) ){
    predict_private(XXs,ZZs);
  }

  // Prediction of statistical data
  if ( (std) || (ei) || (cdf) ){

    if (std) std->fill(-SGTELIB::INF);
    else std = new SGTELIB::Matrix("std",pxx,_m);

    if (ei)   ei->fill(-SGTELIB::INF);
    if (cdf) cdf->fill(-SGTELIB::INF);

    // Use distance to closest as std
    SGTELIB::Matrix dtc = _trainingset.get_distance_to_closest(XXs);
    dtc.set_name("dtc");
    compute_metric_rmse();

    for (j=0 ; j<_m ; j++){
      // Set std
      double s = _metric_rmse[j]; 
      std->set_col( dtc+s , j );

      if (_trainingset.get_bbo(j)==SGTELIB::BBO_OBJ){
        // Compute CDF
        if (cdf){
          for (i=0 ; i<pxx ; i++){
            cdf->set(i,j, normcdf( fs_min , ZZs->get(i,j) , std->get(i,j) ) );
          }
        }
        if (ei){
          for (i=0 ; i<pxx ; i++){
            ei->set(i,j, normei( ZZs->get(i,j) , std->get(i,j) , fs_min ) );
          }
        }
      }// END CASE OBJ
      else if (_trainingset.get_bbo(j)==SGTELIB::BBO_CON){
        // Compute CDF
        if (cdf){
          // Scaled Feasibility Threshold
          double cs = _trainingset.Z_scale(0.0,j);
          for (i=0 ; i<pxx ; i++){
            cdf->set(i,j, normcdf( cs , ZZs->get(i,j) , std->get(i,j) ) );
          }
        }
      }// END CASE CON

    }// End for j

  }
}//






/*--------------------------------------*/
/*               predict                */
/*--------------------------------------*/
void SGTELIB::Surrogate::predict ( const SGTELIB::Matrix & XX ,
                                         SGTELIB::Matrix * ZZ ) {

  check_ready(__FILE__,__FUNCTION__,__LINE__);



  // Check the number of columns in XX
  if (XX.get_nb_cols() != _n){
    display(std::cout); 
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                 "predict(): dimension error" );
  }
  *ZZ = SGTELIB::Matrix("ZZ",XX.get_nb_rows(),_m);

  // Scale the input
  SGTELIB::Matrix XXs(XX);
  _trainingset.X_scale(XXs);


  // Call the private prediction with normalize input XXs
  predict_private( XXs , ZZ );
  #ifdef SGTELIB_DEBUG
    if (ZZ->has_nan()){
      display(std::cout); 
      throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                   "predict(): ZZ has nan" );
    }
  #endif

  // UnScale the output
  _trainingset.Z_unscale(ZZ);

}//

/*--------------------------------------*/
/*       get metric (general)           */
/*--------------------------------------*/
double SGTELIB::Surrogate::get_metric (SGTELIB::metric_t mt , int j){

  // Check dimension
  if ( (j<0) || (j>_m) ){
    display(std::cout); 
    std::cout << "j = "<< j << "\n";
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                 "get_metric(): dimension error" );
  }

  // If the model is not ready, return +INF
  if ( ! _ready){ 
    #ifdef SGTELIB_DEBUG
      std::cout << get_string() << " is not ready => _metric = +INF\n";
    #endif
    return SGTELIB::INF; 
  }

  double m;
  switch(mt){
    case SGTELIB::METRIC_EMAX :
      compute_metric_emax();
      m = _trainingset.ZE_unscale( _metric_emax[j] , j ); 
      break;
    case SGTELIB::METRIC_EMAXCV : 
      compute_metric_emaxcv();
      m = _trainingset.ZE_unscale( _metric_emaxcv[j] , j ); 
      break;
    case SGTELIB::METRIC_RMSE : 
      compute_metric_rmse();
      m = _trainingset.ZE_unscale( _metric_rmse[j] , j ); 
      break;
    case SGTELIB::METRIC_RMSECV: 
      compute_metric_rmsecv();
      m = _trainingset.ZE_unscale( _metric_rmsecv[j] , j ); 
      break;
    case SGTELIB::METRIC_ARMSE : 
      compute_metric_armse();
      m = _metric_armse; 
      break;
    case SGTELIB::METRIC_ARMSECV : 
      compute_metric_armsecv();
      m = _metric_armsecv; 
      break;
    case SGTELIB::METRIC_OE :
      compute_metric_oe();
      m = _metric_oe[j];  
      break;
    case SGTELIB::METRIC_OECV : 
      compute_metric_oecv();
      m = _metric_oecv[j];  
      break;
    case SGTELIB::METRIC_LINV : 
      compute_metric_linv();
      m = _metric_linv[j];  
      break;
    case SGTELIB::METRIC_AOE : 
      compute_metric_aoe();
      m = _metric_aoe; 
      break;
    case SGTELIB::METRIC_AOECV : 
      compute_metric_aoecv();
      m = _metric_aoecv; 
      break;
    case SGTELIB::METRIC_EFIOE : 
      compute_metric_efioe();
      m = _metric_efioe; 
      break;
    case SGTELIB::METRIC_EFIOECV : 
      compute_metric_efioecv();
      m = _metric_efioecv; 
      break;
    default:
      throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
         "get_metric(): unknown metric" );
  }

  if (isnan(m)    ){ m = SGTELIB::INF; }
  if (m < -EPSILON){ m = SGTELIB::INF; }
  if (m <= 0.0    ){ m = 0.0; }
  return m;
}//


/*---------------------------------------*/
/*       compute matrix Zhs              */
/* Zhs is the prediction on the training */
/* points                                */
/*---------------------------------------*/
const SGTELIB::Matrix * SGTELIB::Surrogate::get_matrix_Zhs (void){
  if ( ! _Zhs){
    check_ready(__FILE__,__FUNCTION__,__LINE__);

    //#ifdef SGTELIB_DEBUG
    //#endif
    // Init
    _Zhs = new SGTELIB::Matrix("Zhs",_p,_m);
    //call the predict function on the training points
    predict_private (get_matrix_Xs(),_Zhs);
    _Zhs->replace_nan(+INF);
    _Zhs->set_name("Zhs");
  }
  return _Zhs;
}//


/*--------------------------------------*/
/*       compute matrix Shs             */
/*  (Compute the predictive std)        */
/*--------------------------------------*/
const SGTELIB::Matrix * SGTELIB::Surrogate::get_matrix_Shs (void){
  if ( ! _Shs){
    check_ready(__FILE__,__FUNCTION__,__LINE__);

    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _Shs\n";
    #endif
    // Init
    _Shs = new SGTELIB::Matrix("Shs",_p,_m);
    //call the predict function on the training points
    predict_private (get_matrix_Xs(),NULL,_Shs,NULL,NULL);
    _Shs->replace_nan(+INF);
    _Shs->set_name("Shs");
  }
  return _Shs;
}//

// If no specific method is defined, consider Svs = Shs.
const SGTELIB::Matrix * SGTELIB::Surrogate::get_matrix_Svs (void){
  if ( ! _Svs){
    _Svs = new SGTELIB::Matrix("Svs",_p,_m);
    const SGTELIB::Matrix Ds = _trainingset.get_matrix_Ds();
    for (int i=0 ; i<_p ; i++){
      double dmin = +INF;
      for (int j=0 ; j<_p ; j++){
        if (i!=j){
          dmin = std::min(dmin,Ds.get(i,j));
        }
      }
      _Svs->set_row(dmin,i);
    }
  }
  return _Svs;
}//



/*--------------------------------------*/
/*       get_Xs                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Xs (void){
  _trainingset.build(); 
  return _trainingset.get_matrix_Xs().get_rows(_selected_points);
}//


/*--------------------------------------*/
/*       get_Zs                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Zs (void){
  _trainingset.build(); 
  return _trainingset.get_matrix_Zs().get_rows(_selected_points);
}//


/*--------------------------------------*/
/*       get_Ds                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Ds (void){
  _trainingset.build(); 
  return _trainingset.get_matrix_Ds().get( _selected_points , _selected_points );
}//


/*--------------------------------------*/
/*       get_Zv                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Zv (void){
  // Return unscaled matrix Zv
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  SGTELIB::Matrix Zv (*get_matrix_Zvs()); // Get scaled matrix
  _trainingset.Z_unscale(&Zv); // Unscale
  return Zv; // Return unscaled
}//


/*--------------------------------------*/
/*       get_Zh                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Zh (void){
  // Return unscaled matrix Zh
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  SGTELIB::Matrix Zh (*get_matrix_Zhs()); // Get scaled matrix
  _trainingset.Z_unscale(&Zh); // Unscale
  return Zh; // Return unscaled
}//


/*--------------------------------------*/
/*       get_Sh                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Sh (void){
  // Return unscaled matrix Shs
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  SGTELIB::Matrix Sh = (*get_matrix_Shs());
  _trainingset.ZE_unscale(&Sh); // Unscale
  return Sh; // Return unscaled
}//

/*--------------------------------------*/
/*       get_Sh                         */
/*--------------------------------------*/
const SGTELIB::Matrix SGTELIB::Surrogate::get_matrix_Sv (void){
  // Return unscaled matrix Zh
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  SGTELIB::Matrix Sv (*get_matrix_Svs()); // Get scaled matrix
  _trainingset.ZE_unscale(&Sv); // Unscale
  return Sv; // Return unscaled
}//


/*--------------------------------------*/
/*       compute rmsecv                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_rmsecv (void){
  check_ready();
  if ( ! _metric_rmsecv){
    // Init
    _metric_rmsecv = new double [_m];

    // Call to the method of the derivated class to 
    // compute Zv

    int i,j;
    double e;
    const SGTELIB::Matrix Zs = get_matrix_Zs();
    const SGTELIB::Matrix * Zvs = get_matrix_Zvs();

    // Loop on the outputs
    for (j=0 ; j<_m ; j++){
      // Compute the error for output j
      e = 0;
      for (i=0 ; i<_p ; i++){
        e += pow(Zs.get(i,j)-Zvs->get(i,j),2);
      }
      _metric_rmsecv[j] = sqrt(e/_p);
    }
  }
}//

/*--------------------------------------*/
/*       compute emax                   */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_emax (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_emax){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_emax\n";
    #endif
    // Init
    _metric_emax = new double [_m];

    int i,j;
    double e;
    const SGTELIB::Matrix Zs = get_matrix_Zs();
    const SGTELIB::Matrix * Zhs = get_matrix_Zhs();
    // Loop on the outputs
    for (j=0 ; j<_m ; j++){
      // Compute the error for output j
      e = 0;
      for (i=0 ; i<_p ; i++){
        e = std::max( e , fabs( Zs.get(i,j)-Zhs->get(i,j) ) );
      }
      _metric_emax[j] = e;
    }
  }

  #ifdef SGTELIB_DEBUG
    std::cout << "metric_emax: " ;
    for (int j=0 ; j<_m ; j++){
      std::cout << _metric_emax[j] << " ";
    }
    std::cout << "\n";
  #endif
}//


/*--------------------------------------*/
/*       compute emaxcv                 */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_emaxcv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_emaxcv){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_emaxcv\n";
    #endif
    // Init
    _metric_emaxcv = new double [_m];

    int i,j;
    double e;
    const SGTELIB::Matrix Zs = get_matrix_Zs();
    const SGTELIB::Matrix * Zvs = get_matrix_Zvs();
    // Loop on the outputs
    for (j=0 ; j<_m ; j++){
      // Compute the error for output j
      e = 0;
      for (i=0 ; i<_p ; i++){
        e = std::max( e , fabs( Zs.get(i,j)-Zvs->get(i,j) ) );
      }
      _metric_emaxcv[j] = e;
    }
  }

  #ifdef SGTELIB_DEBUG
    std::cout << "metric_emaxcv: " ;
    for (int j=0 ; j<_m ; j++){
      std::cout << _metric_emaxcv[j] << " ";
    }
    std::cout << "\n";
  #endif

}//

/*--------------------------------------*/
/*       compute rmse                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_rmse (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_rmse){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_rmse\n";
    #endif
    // Init
    _metric_rmse = new double [_m];

    int i,j;
    double e;
    const SGTELIB::Matrix Zs = get_matrix_Zs();
    const SGTELIB::Matrix * Zhs = get_matrix_Zhs();
    // Loop on the outputs
    for (j=0 ; j<_m ; j++){
      // Compute the error for output j
      e = 0;
      for (i=0 ; i<_p ; i++){
        e += pow(Zs.get(i,j)-Zhs->get(i,j),2);
      }
      _metric_rmse[j] = sqrt(e/_p);
    }
  }

  #ifdef SGTELIB_DEBUG
    std::cout << "metric_rmse: " ;
    for (int j=0 ; j<_m ; j++){
      std::cout << _metric_rmse[j] << " ";
    }
    std::cout << "\n";
  #endif

}//




/*--------------------------------------*/
/*       compute oe                   */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_oe (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_oe){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_oe\n";
    #endif
    // Init
    _metric_oe = new double [_m];
    // Compute the prediction on the training points
    const SGTELIB::Matrix * Zhs = get_matrix_Zhs();
    // Compute the order-efficiency metric using the matrix Zh
    // nb: oe   => use matrix _Z
    //     oecv => use matrix _Zv 
    compute_order_error(Zhs,_metric_oe);
  }
}//

/*--------------------------------------*/
/*       compute oecv                   */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_oecv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_oecv){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_oecv\n";
    #endif
    // Init
    _metric_oecv = new double [_m];
    // Compute the prediction on the training points
    const SGTELIB::Matrix * Zvs = get_matrix_Zvs();
    // Compute the order-efficiency metric using the matrix Zh
    // nb: oe   => use matrix _Z
    //     oecv => use matrix _Zv 
    compute_order_error(Zvs,_metric_oecv);
  }
}//



/*--------------------------------------*/
/*       compute aoe                    */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_aoe (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_aoe<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_aoe\n";
    #endif
    // Compute the prediction on the training points
    const SGTELIB::Matrix * Zhs = get_matrix_Zhs();
    _metric_aoe = compute_aggregate_order_error(Zhs);
  }
}//


/*--------------------------------------*/
/*       compute aoecv                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_aoecv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_aoecv<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_aoecv\n";
    #endif
    // Compute the prediction on the training points
    const SGTELIB::Matrix * Zvs = get_matrix_Zvs();
    _metric_aoecv = compute_aggregate_order_error(Zvs);
  }
}//


/*--------------------------------------*/
/*       compute efioe                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_efioe (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_efioe<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_efioe\n";
    #endif
    SGTELIB::Matrix * EFI = new SGTELIB::Matrix("EFI",_p,_m);
    EFI->fill(-1);
    EFI->set_col(compute_efi(*get_matrix_Zhs(),*get_matrix_Shs()),_trainingset.get_j_obj());
    _metric_efioecv = compute_aggregate_order_error(EFI);
    delete EFI;
  }
}//

/*--------------------------------------*/
/*       compute efioecv                */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_efioecv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_efioecv<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_efioe\n";
    #endif
    SGTELIB::Matrix * EFI = new SGTELIB::Matrix("EFI",_p,_m);
    EFI->fill(-1);
    EFI->set_col(compute_efi(*get_matrix_Zvs(),*get_matrix_Svs()),_trainingset.get_j_obj());
    _metric_efioecv = compute_aggregate_order_error(EFI);
    delete EFI;
  }
}//


/*----------------------------------------------------------*/
/*     compute EFI from the predictive mean and std         */
/*----------------------------------------------------------*/
SGTELIB::Matrix SGTELIB::Surrogate::compute_efi( const SGTELIB::Matrix Zs,
                                                 const SGTELIB::Matrix Ss  ){

  const int p = Zs.get_nb_rows();
  if (Zs.get_nb_cols()!=_m) throw SGTELIB::Exception ( __FILE__ , __LINE__ ,"Unconsistent nb of cols" );

  const SGTELIB::Matrix Z = _trainingset.Z_unscale(Zs);
  const SGTELIB::Matrix S = _trainingset.ZE_unscale(Ss);
  const double fmin = _trainingset.get_f_min();

  SGTELIB::Matrix EFI ("EFI",p,1);
  EFI.fill(1.0);
  double v;

  for (int j=0 ; j<_m ; j++){
    if (_trainingset.get_bbo(j)==SGTELIB::BBO_OBJ){
      for (int i=0 ; i<p ; i++){
        v = SGTELIB::normei( Z.get(i,j) , S.get(i,j) , fmin );
        EFI.product(i,0,v);
      }
    }
    if (_trainingset.get_bbo(j)==SGTELIB::BBO_CON){
      for (int i=0 ; i<p ; i++){
        v = SGTELIB::normcdf( 0.0 , Z.get(i,j) , S.get(i,j) );
        EFI.product(i,0,v);
      }
    }
  }// end loop on j

  return EFI;

}//





/*--------------------------------------*/
/*       compute armse                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_armse (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_armse<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_armse\n";
    #endif
    compute_metric_rmse();
    _metric_armse = 0;
    for (int j=0 ; j<_m ; j++) _metric_armse += _metric_rmse[j]; 
  }
}//


/*--------------------------------------*/
/*       compute armsecv                  */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_armsecv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if (_metric_armsecv<0){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_armsecv\n";
    #endif
    compute_metric_rmsecv();
    _metric_armsecv = 0;
    for (int j=0 ; j<_m ; j++) _metric_armsecv += _metric_rmsecv[j]; 
  }
}//

/*--------------------------------------*/
/*       compute linv                   */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_metric_linv (void){
  check_ready(__FILE__,__FUNCTION__,__LINE__);
  if ( ! _metric_linv){
    #ifdef SGTELIB_DEBUG
      std::cout << "Compute _metric_linv\n";
    #endif
    // Init
    _metric_linv = new double [_m];

    // Compute the prediction on the training points
    const SGTELIB::Matrix * Zhs = get_matrix_Zhs();
    const SGTELIB::Matrix * Shs = get_matrix_Shs();
    // True values
    const SGTELIB::Matrix Zs = get_matrix_Zs();
    double s,dz;
    double linv;
    // TODO : improve the behavior of linv for very small s.
    for (int j=0 ; j<_m ; j++){
      if (_trainingset.get_bbo(j)!=SGTELIB::BBO_DUM){
        linv = 0;
        for (int i=0 ; i<_p ; i++){
          dz = Zhs->get(i,j)-Zs.get(i,j);
          s  = Shs->get(i,j);
          s = std::max(s ,EPSILON);   
          dz= std::max(dz,EPSILON); 
          linv += -log(s) - pow(dz/s,2)/2;
        }
        linv /= _p; // normalization by the number of points
        linv -= 0.5*log(2*3.141592654); // add the normal pdf constant
        // Add this point, we have log(prod g)/p
        linv = exp(-linv);
        _metric_linv[j] = linv;
      }
      else{
        _metric_linv[j] = -SGTELIB::INF;
      }
    }
  }

}//



/*--------------------------------------*/
/*       compute order efficiency       */
/*--------------------------------------*/
void SGTELIB::Surrogate::compute_order_error (const SGTELIB::Matrix * const Zpred , 
                                              double * m                   ){

  check_ready(__FILE__,__FUNCTION__,__LINE__);
  // Compute the order-efficiency metric by comparing the 
  // values of - _Zs (in the trainingset)
  //           - Zpred (input of this function)
  // Put the results in "m" (output of this function)

  if ( ! m){
    display(std::cout); 
    throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                   "compute_order_error(): m is NULL" );
  }

  int nb_fail;
  const SGTELIB::Matrix Zs = get_matrix_Zs();
  
  for (int j=0 ; j<_m ; j++){
    switch (_trainingset.get_bbo(j)){
    //===============================================//
    case SGTELIB::BBO_OBJ:
      double z1,z1h,z2,z2h;
      nb_fail = 0;
      for (int i1=0 ; i1<_p ; i1++){
        z1 = Zs.get(i1,j);
        z1h = Zpred->get(i1,j);
        for (int i2=0 ; i2<_p ; i2++){
          z2 = Zs.get(i2,j);
          z2h = Zpred->get(i2,j);
          if ( (z1-z2<0)^(z1h-z2h<0) ) nb_fail++;
        }
      }
      m[j] = double(nb_fail)/double(_p*_p);
      break;
    //===============================================//
    case SGTELIB::BBO_CON:
      nb_fail = 0;
      double z,zh;
      for (int i=0 ; i<_p ; i++){
        z = Zs.get(i,j);
        zh = Zpred->get(i,j);
        if ( (z<0)^(zh<0) ) nb_fail++;
      }

      m[j] = double(nb_fail)/double(_p);
      break;
    //===============================================//
    case SGTELIB::BBO_DUM:
      m[j] = -1.0;
      break;
    //===============================================//
    default:
      display(std::cout); 
      throw SGTELIB::Exception ( __FILE__ , __LINE__ ,"Undefined type" );
    //===============================================//
    }// end switch
  }// end loop on j
}//


/*--------------------------------------*/
/*       compute order efficiency       */
/*--------------------------------------*/
double SGTELIB::Surrogate::compute_aggregate_order_error (const SGTELIB::Matrix * const Zpred){

  check_ready(__FILE__,__FUNCTION__,__LINE__);

  const SGTELIB::Matrix Zs = get_matrix_Zs();

  // Build f1,h1,f2 and h2.
  // f1 and h1 are the real data
  // f2 and h2 are the surrogate.

  SGTELIB::Matrix fhr ("fhr",_p,2);
  SGTELIB::Matrix fhs ("fhs",_p,2);
  fhr.fill(0.0);
  fhs.fill(0.0);
  int i,j;
  for (j=0 ; j<_m ; j++){
    switch (_trainingset.get_bbo(j)){
    //===============================================//
    case SGTELIB::BBO_OBJ:
      fhr.set_col( Zs.get_col(j) , 0 );
      fhs.set_col( Zpred->get_col(j) , 0 );
      break;
    //===============================================//
    case SGTELIB::BBO_CON:
      for (i=0 ; i<_p ; i++){
        double d;
        d = Zs.get(i,j);
        if (d>0) fhr.add(i,1,d*d);
        d = Zpred->get(i,j);
        if (d>0) fhs.add(i,1,d*d);
      }
      break;
    //===============================================//
    case SGTELIB::BBO_DUM:
      break;
    //===============================================//
    default:
      display(std::cout); 
      throw SGTELIB::Exception ( __FILE__ , __LINE__ ,"Undefined type" );
    //===============================================//
    }// end switch
  }// end loop on j

  int e = 0;
  int i1,i2;
  double hr1,hr2,hs1,hs2,fr1,fr2,fs1,fs2;
  bool inf_r,inf_s;
  // i1 and i2 are the indexes of the two points that are compared.
  // fr1 and hr1 (resp. fr2 and hr2) are the real values of f and r for these points.
  // fs1 and hs1 (resp. fs2 and hs2) are the surrogate (or CV) values.
  for (i1=0 ; i1<_p ; i1++){
    fr1 = fhr.get(i1,0);
    hr1 = fhr.get(i1,1);
    fs1 = fhs.get(i1,0);
    hs1 = fhs.get(i1,1);
    for (i2=0 ; i2<_p ; i2++){
      fr2 = fhr.get(i2,0);
      hr2 = fhr.get(i2,1);
      fs2 = fhs.get(i2,0);
      hs2 = fhs.get(i2,1);
      // Compute the order for real (r) data and for surrogate (s) model
      inf_r = ( (hr1<hr2) | ( (hr1==hr2) & (fr1<fr2) ) );
      inf_s = ( (hs1<hs2) | ( (hs1==hs2) & (fs1<fs2) ) );
      // If they don't agree, increment e. (Note that ^ is the xor operator)
      if (inf_r ^ inf_s) e++;
    }
  }
  return double(e)/double(_p*_p);

}//

/*--------------------------------------*/
/*       get_exclusion_area_penalty     */
/*--------------------------------------*/
SGTELIB::Matrix SGTELIB::Surrogate::get_exclusion_area_penalty ( const SGTELIB::Matrix & XX , const double tc ) const{
  // Scale the input
  SGTELIB::Matrix XXs(XX);
  XXs.set_name("XXs");
  _trainingset.X_scale(XXs);
  return _trainingset.get_exclusion_area_penalty ( XXs , tc );
}//


/*--------------------------------------*/
/*       get_distance_to_closest        */
/*--------------------------------------*/
SGTELIB::Matrix SGTELIB::Surrogate::get_distance_to_closest ( const SGTELIB::Matrix & XX ) const{
  // Scale the input
  SGTELIB::Matrix XXs(XX);
  XXs.set_name("XXs");
  _trainingset.X_scale(XXs);
  return _trainingset.get_distance_to_closest ( XXs );
}//




/*--------------------------------------*/
/*  optimize model parameters           */
/*--------------------------------------*/
bool SGTELIB::Surrogate::optimize_parameters ( void ) {


  // Number of parameters to optimize
  const int N = _param.get_nb_parameter_optimization();
  // Budget
  int budget = N*_param.get_budget();

  int i,j,k;
  double d;
  const bool display = false;
  if (display){
    std::cout << "Begin parameter optimization\n";
    std::cout << "Metric: " << SGTELIB::metric_type_to_str(_param.get_metric_type()) << "\n";
  }

  

  //-----------------------------------------
  // Bounds, Scaling and domain
  //-----------------------------------------
  SGTELIB::Matrix lb("lb",1,N);
  SGTELIB::Matrix ub("ub",1,N);
  SGTELIB::Matrix scaling ("scaling",1,N);
  bool * logscale = new bool [N];
  SGTELIB::param_domain_t * domain = new SGTELIB::param_domain_t[N];

  _param.get_x_bounds ( &lb , &ub , domain , logscale );

  for (i=0 ; i<N ; i++){
    if (domain[i]==SGTELIB::PARAM_DOMAIN_CONTINUOUS){
      if (logscale[i]) d = 1;
      else d = (ub[i]-lb[i])/5;
      scaling.set(0,i,d);
      if (d<EPSILON) throw SGTELIB::Exception ( __FILE__ , __LINE__ ,"Bad scaling." );
    }
    else if (domain[i]==SGTELIB::PARAM_DOMAIN_CAT){
      scaling.set(0,i,ub[i]-lb[i]);
    }
    else{
      scaling.set(0,i,1);
    }
  }

  if (display){
    std::cout << "Model: " << get_short_string() << "\n";
    std::cout << "lb: [ ";
    for (i=0 ; i<N ; i++) std::cout << lb[i] << " ";
    std::cout << "]\n";
    std::cout << "ub: [ ";
    for (i=0 ; i<N ; i++) std::cout << ub[i] << " ";
    std::cout << "]\n";
    std::cout << "scaling: [ ";
    for (i=0 ; i<N ; i++){
      std::cout << scaling[i];
      if (logscale[i]) std::cout << "(log)";
      std::cout << " ";
    }
    std::cout << "]\n";
  }

  // Build set of starting points
  const int nx0 = 1+budget/10;
  SGTELIB::Matrix X0 ("X0",nx0,N);
  X0.set_row(_param.get_x(),0);
  for (j=0 ; j<N ; j++){
    double lbj = lb[j];
    double ubj = ub[j];
    for (i=1 ; i<nx0 ; i++){ // nb: Skip the first row of X0
      d = uniform_rand();
      if (logscale[j]) d = lb[j] * pow(ubj/lbj,d);
      else d = lbj + (ubj-lbj)*d;
      X0.set(i,j,d);
    } 
  }
  
  //---------------------------------------------
  // Budget, poll size, success and objectives
  //---------------------------------------------

  SGTELIB::Matrix xtry ("xtry",1,N);
  double fmin = +INF;
  double pmin = +INF;
  double ftry, ptry;
  bool success;
  double psize = 0.5;
  SGTELIB::Matrix POLL;
  SGTELIB::Matrix xmin = X0.get_row(0);

  // Init cache of evaluated points
  SGTELIB::Matrix CACHE ("CACHE",0,N);
  bool cache_hit;

  //------------------------
  // LOOP
  //------------------------
  int iter=0;
  while (budget>0){

    success = false;

    if (display){
      std::cout << "=================================================\n";
      std::cout << "Budget: " << budget  << "\n";
      // Display best solution
      std::cout << "\nCurrent xmin:\n";
      std::cout << "X=[ " ;
      for (j=0 ; j<N ; j++) std::cout << xmin[j] << " ";
      std::cout << "] => " << fmin << " / " << pmin <<  "\n\n";
    }

    if (iter){
      // Create POLL candidates
      POLL = SGTELIB::Matrix::get_poll_directions(scaling,domain,psize);
      //POLL.display(std::cout);
      for (i=0 ; i<POLL.get_nb_rows() ; i++){
        for (j=0 ; j<N ; j++){
          // Add poll directions to poll center
          d = xmin[j];
          if (logscale[j]) d *= pow(4.0,POLL.get(i,j));  //exp(POLL.get(i,j));
          else             d += POLL.get(i,j);
          xtry.set(0,j,d);    
        }// End build candidate
        POLL.set_row(xtry,i);
      } // End Create POLL
      POLL.set_name("POLL-CANDIDATES");
      //POLL.display(std::cout);
    }
    else{
      // If iter==0, then evaluate starting points
      POLL = X0;
    }

    // Evaluate POLL
    for (i=0 ; i<POLL.get_nb_rows() ; i++){

      // Candidate
      xtry = POLL.get_row(i);
      xtry.set_name("xtry");

      // Display candidate
      if (display){
        if (iter) std::cout << "X = [ " ;
        else std::cout << "X0= [ " ;
        for (j=0 ; j<N ; j++) std::cout << xtry[j] << " ";
        std::cout << "] => ";
      }

      // Snap to bounds
      for (j=0 ; j<N ; j++){
        d = xtry[j];
        // Snap to bounds
        double lbj = lb[j];
        double ubj = ub[j];
        switch (domain[j]){
          case SGTELIB::PARAM_DOMAIN_CONTINUOUS:
            if (d<lbj) d = lbj;
            if (d>ubj) d = ubj;
            break;
          case SGTELIB::PARAM_DOMAIN_INTEGER:
            d = double(round(d));
            if (d<lbj) d=lbj;
            if (d>ubj) d=ubj;
            break;
          case SGTELIB::PARAM_DOMAIN_CAT:
            k = round(d);
            while (k>ubj) k-=int(ubj-lbj);
            while (k<lbj) k+=int(ubj-lbj);
            d = double(k);
            break;
          case SGTELIB::PARAM_DOMAIN_BOOL:
            d = (d>1/2)?1.0:0.0;
            break;
          case SGTELIB::PARAM_DOMAIN_MISC:
            throw SGTELIB::Exception ( __FILE__ , __LINE__ ,"Invalid variable domain!" );
            break;
        }
        xtry.set(0,j,d);    
      }

      // Check Cache
      cache_hit = (CACHE.find_row(xtry)!=-1);
      if (cache_hit){
        if (display) std::cout << "Cache hit\n";
      }
      else{
        // ---------------------------------
        // EVALUATION 
        // ---------------------------------
        _param.set_x(xtry);
        _param.check();
        ftry = eval_objective();
        ptry = _param.get_x_penalty();
        budget--;
        CACHE.add_rows(xtry);

        // Display f
        if (display){
          if (ftry>=+INF) std::cout << "+inf" ;
          else std::cout << ftry;
          std::cout << " / " ;
          if (ptry>=+INF) std::cout << "+inf" ;
          else std::cout << ptry;
        }

        // Check for success for each objective
        if ( (ftry<fmin) || ((ftry==fmin) && (ptry<pmin)) ){
          if (display) std::cout << "(!)";
          xmin = xtry;
          fmin = ftry;
          pmin = ptry;
          success = true;
        }
        if (display) std::cout << "\n";
      } // End Evaluation (i.e. No Cache Hit)

      if ( (iter) && (success) ) break;

    }// END LOOP ON POLL (for i...)

    if (iter){
      // Update poll size
      if (success) psize*=2;
      else psize/=2;
    }
    iter++;

    // Check convergence
    if (psize<1e-6) break;
    if (budget<=0) break;

  }// End of optimization

  // Set param to optimal value
  _param.set_x(xmin);
  _param.check();

  fmin = eval_objective();
  /*
  _param.display(std::cout);
  std::cout << "fmin = " << fmin << "\n";
  std::cout << "=================================\n";
  */
  if (display){
    _param.display(std::cout);
    std::cout << "End parameter optimization\n";
    std::cout << "=================================\n";
  }

  // Check for Nan
  if (xmin.has_nan() || xmin.has_inf()) return false;

  delete [] logscale;
  delete [] domain;

  // Return success
  return true;

}//


/*--------------------------------------*/
/*    Evaluate a set of parameters      */
/*--------------------------------------*/
double SGTELIB::Surrogate::eval_objective ( void ){

  //std::cout << "Eval obj...\n";
  reset_metrics();

  // Build model
  bool ok = build_private();
  if ( ! ok) return +INF;

  // Compute metric
  const SGTELIB::metric_t mt = _param.get_metric_type();

  double metric = 0;
  if (SGTELIB::metric_multiple_obj(mt)){
    for (int i=0 ; i<_m ; i++) metric += get_metric(mt,i);
  }
  else{
    metric = get_metric(mt,0);
  }

  if ( isnan(metric) ) return +INF;
  if ( isinf(metric) ) return +INF;
  return metric;

}//









