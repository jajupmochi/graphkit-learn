/*-------------------------------------------------------------------------------------*/
/*  sgtelib - A surrogate model library for derivative-free optimization               */
/*  Version 2.0.1                                                                      */
/*                                                                                     */
/*  Copyright (C) 2012-2016  Sebastien Le Digabel - Ecole Polytechnique, Montreal      */ 
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
/**
 \file   Sgtelib_Model_Manager.cpp
 \brief  Handle the sgtelib_model model for the search and the eval_sort.
 \author Bastien Talgorn
 \date   2013-04-25
 \see    Sgtelib_Model_Search.cpp
 */
#include "Sgtelib_Model_Manager.hpp"
#include "Evaluator_Control.hpp"


/*---------------------------------------------------------------*/
/*                    constructor                                */
/*---------------------------------------------------------------*/
NOMAD::Sgtelib_Model_Manager::Sgtelib_Model_Manager ( NOMAD::Parameters & p ,
                                                     NOMAD::Evaluator_Control * ev_control ) :
_p            ( p           ),
_ev_control   ( ev_control  ),
_highest_tag  ( -1          ),
_ready        ( false       )
{
    
    
    // Initialization of model bounds
    _model_lb = NOMAD::Point( _p.get_dimension(),+NOMAD::INF );
    _model_ub = NOMAD::Point( _p.get_dimension(),-NOMAD::INF );
    
    // Initialisation of best objective
    _found_feasible = false;
    
    // Initialization of statistics
    _search_pfi_max = 0.0;
    _search_efi_max = 0.0;
    _search_obj_min = +NOMAD::INF;
    
    
    // EXTERN SGTE
    if ( _p.get_SGTELIB_MODEL_FORMULATION()==NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
        return;
    
    // Check
    if ( (_p.get_SGTELIB_MODEL_FORMULATION()==NOMAD::SGTELIB_MODEL_FORMULATION_FS) ||
        (_p.get_SGTELIB_MODEL_FORMULATION()==NOMAD::SGTELIB_MODEL_FORMULATION_EIS) )
    {
        if ( ! (_p.get_SGTELIB_MODEL_FEASIBILITY() == NOMAD::SGTELIB_MODEL_FEASIBILITY_C) )
        {
            std::cout << "ERROR : Formulations FS and EIS can only be used with FeasibilityMethod C" << std::endl;
            throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                                      "Sgtelib_Model_Manager: SGTELIB_MODEL_FEASIBILITY not valid." );
        }
    }
    
    // Count the number of constraints
    int nb_constraints = 0;
    const std::vector<NOMAD::bb_output_type> bbot = _p.get_bb_output_type();
    for ( int j=0 ; j<_p.get_bb_nb_outputs() ; j++)
    {
        if ( bbot_is_constraint(bbot[j]) )
            nb_constraints++;
    }
    
    // Define how many models to build
    switch ( _p.get_SGTELIB_MODEL_FEASIBILITY() )
    {
        case NOMAD::SGTELIB_MODEL_FEASIBILITY_C:
            _nb_models = 1+nb_constraints;
            break;
        case NOMAD::SGTELIB_MODEL_FEASIBILITY_H:
        case NOMAD::SGTELIB_MODEL_FEASIBILITY_B:
        case NOMAD::SGTELIB_MODEL_FEASIBILITY_M:
            _nb_models = 2;
            break;
        case NOMAD::SGTELIB_MODEL_FEASIBILITY_UNDEFINED:
            std::cout<< "UNDEFINED_SGTELIB_MODEL_FEASIBILITY" << std::endl;
            break;
    }
    
    // Init the TrainingSet
    SGTELIB::Matrix empty_X ( "empty_X" , 0 , _p.get_dimension() );
    SGTELIB::Matrix empty_Z ( "empty_Z" , 0 , _nb_models );
    _trainingset = new SGTELIB::TrainingSet( empty_X , empty_Z );
    
    // Build the Sgtelib Model
    std::string model_definition = _p.get_SGTELIB_MODEL_DEFINITION();
    _model = SGTELIB::Surrogate_Factory( *_trainingset , model_definition );
    
}// end of constructor





/*---------------------------------------------------------------*/
/*                    reset                                      */
/*---------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::reset(void)
{
    _highest_tag = -1;
    
    if ( _model )
    {
        SGTELIB::surrogate_delete( _model );
        _model = NULL;
    }
    
    if ( _trainingset )
    {
        delete _trainingset;
        _trainingset = NULL;
    }
    
    _ready = false;
}

/*---------------------------------------------------------------*/
/*                      is_ready                                 */
/* Return a boolean to know if the sgtelib_model model has been  */
/* build and can be called for predictions.                      */
/*---------------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Manager::is_ready(void)
{
    
    if ( _ready ) 
        return true;
    
    // EXTERN SGTE
    if ( _p.get_SGTELIB_MODEL_FORMULATION()==NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
    {
        _ready = true;
        return _ready;
    }
    if ( !_trainingset )
        throw NOMAD::Exception ( __FILE__ , __LINE__ ,
                                "Sgtelib_Model_Manager::is_ready : no training set!" );
    
    if ( ! _trainingset->is_ready() )
        return false;
    
    const int pvar = _trainingset->get_pvar();
    _ready = ( ( _highest_tag != -1 ) && ( _model->is_ready() ) && ( pvar > 10 ) );
    return _ready;
}


/*---------------------------------------------------------------*/
/*               Reset Search Stats                              */
/*---------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::reset_search_stats(void)
{
    _search_pfi_max = 0;
    _search_efi_max = 0;
    _search_obj_min = NOMAD::INF;
}//

void NOMAD::Sgtelib_Model_Manager::write_search_stats(void) const
{
    // Write the point in memory file
    ofstream memory;
    memory.open ("memory.txt", ios::app );
    if ( memory.is_open() )
    {
        memory.precision(24);
        memory << "#SEARCH_STATS" << std::endl;
        memory << _search_pfi_max << " "
        << _search_efi_max << " "
        << _search_obj_min << std::endl;
        memory.close();
    }
}//



/*---------------------------------------------------------------*/
/*                   info                                        */
/*---------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::info(void)
{
    std::cout << "  #===================================================== #" << std::endl;
    std::cout << "Sgtelib_Model_Manager::info" << std::endl ;
    std::cout << "Sgtelib_Model_Manager : " << this << std::endl;
    std::cout << "ev_control : " << _ev_control << std::endl;
    std::cout << "Model : " << _model << std::endl;
    std::cout << "highest_tag : " << _highest_tag << std::endl;
    
    NOMAD::Cache & cache = _ev_control->get_cache();
    std::cout << "Cache size : " << cache.size() << std::endl;
    std::cout << "found_feasible : " << _found_feasible << std::endl; 
    
    // Display of the model's bounds.
    int i;
    int n = _p.get_dimension();
    std::cout << "Model Bounds, lb : ( ";
    for  ( i=0 ; i<n ; i++ )
        std::cout << _model_lb.get_coord(i) << " ";
    
    std::cout << ") , ub : ( ";
    for  ( i=0 ; i<n ; i++ )
        std::cout << _model_ub.get_coord(i) << " ";
    
    std::cout << ")" << std::endl;
    
    std::cout << "Model Ext Bounds, lb : ( ";
    NOMAD::Point ext_lb = get_extended_lb();
    for  ( i=0 ; i<n ; i++ )
        std::cout << ext_lb.get_coord(i) << " ";
    
    std::cout << ") , ub : ( ";
    NOMAD::Point ext_ub = get_extended_ub();
    for  ( i=0 ; i<n ; i++ )
        std::cout << ext_ub.get_coord(i) << " ";
    
    std::cout << ")" << std::endl;
    
    if ( _ready )
        std::cout << "sgtelib_model model is ready" << std::endl;
    else
        std::cout << "sgtelib_model model is NOT ready" << std::endl;
    
    
    std::cout << "  #===================================================== #" << std::endl;
}




/*------------------------------------------------------------------*/
/*                          MODEL UPDATE                            */
/*------------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::update(void)
{
    const NOMAD::Display & out = _p.out();    
    const bool display_update = string_find( _p.get_SGTELIB_MODEL_DISPLAY(),"U") ;
    
    if ( display_update )
        out.open_block("Update sgtelib model");
    
    // EXTERN SGTE
    if ( _p.get_SGTELIB_MODEL_FORMULATION() == NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
    {
        if ( display_update ) out << "FORMULATION: EXTERN." << std::endl;
        return;
    }
    
    NOMAD::Cache & cache = _ev_control->get_cache();
    const std::vector<NOMAD::bb_output_type> bbot = _p.get_bb_output_type();
    // row_X et row_Z sont les matrices à une seule ligne
    // permettant de stocker le point courant
    SGTELIB::Matrix row_X ( "row_X" , 1 , _p.get_dimension() );
    SGTELIB::Matrix row_Z ( "row_Z" , 1 , _nb_models );
    
    // Matrices permettant de stocker tous les points à envoyer vers le model
    SGTELIB::Matrix add_X ( "add_X" , 0 , _p.get_dimension() );
    SGTELIB::Matrix add_Z ( "add_Z" , 0 , _nb_models );
    
    if ( display_update )
        out << "Review of the cache" << std::endl;
    
    // Parcours des points de la cache
    int k;
    NOMAD::Double v;
    int next_highest_tag = _highest_tag;
    const NOMAD::Eval_Point * cur = cache.begin();
    int tag;
    bool valid_point;
    // Parcours de la cache
    while ( cur )
    {
        tag = cur->get_tag() ;
        if ( tag > _highest_tag )
        {
            if ( display_update )
            {
                out << "New Tag : " << tag << std::endl;
                out << "xNew = ( ";
                for ( int i = 0 ; i < _p.get_dimension() ; i++ )
                {
                    out << cur->get_coord(i).value() << " ";
                }
                out << ")";
            }
            
            // Vérification que le point est valide (pas de Nan, pas d'échec, toutes les sorties définies et la fonction cout disponible )
            valid_point = true;
            for ( int j=0 ; j < _p.get_bb_nb_outputs() ; j++)
            {
                if ( ( !cur->get_bb_outputs()[j].is_defined() ) || ( isnan(cur->get_bb_outputs()[j].value()) ) || !cur->is_eval_ok() || !cur->get_f().is_defined() )
                    valid_point = false;
            }
            
            if ( display_update )
            {
                if ( !valid_point ) out << " (not valid) ";
            }
            
            if ( valid_point )
            {
                // X
                for ( int j=0 ; j < _p.get_dimension() ; j++ )
                    row_X.set( 0, j, cur->get_coord(j).value()  );
                
                add_X.add_rows(row_X);
                
                // Objective
                row_Z.set(0,0,cur->get_f().value()); // 1st column: constraint model
                
                // Constraints
                switch ( _p.get_SGTELIB_MODEL_FEASIBILITY() )
                {
                    case NOMAD::SGTELIB_MODEL_FEASIBILITY_C:
                        k=1;
                        for ( int j = 0 ; j < _p.get_bb_nb_outputs() ; j++ )
                        {
                            if ( bbot_is_constraint(bbot[j]) )
                            {
                                row_Z.set( 0, k, cur->get_bb_outputs()[j].value() );
                                k++;
                            }
                        }
                        break;
                        
                    case NOMAD::SGTELIB_MODEL_FEASIBILITY_H:
                        eval_h(cur->get_bb_outputs(),v);
                        row_Z.set(0,1,v.value()); // 2nd column: constraint model
                        break;
                        
                    case NOMAD::SGTELIB_MODEL_FEASIBILITY_B:
                        row_Z.set(0,1,cur->is_feasible(0.)); // 2nd column: constraint model
                        break;
                        
                    case NOMAD::SGTELIB_MODEL_FEASIBILITY_M:
                        v = -NOMAD::INF;
                        for ( int j=0 ; j < _p.get_bb_nb_outputs() ; j++)
                        {
                            if ( bbot_is_constraint( bbot[j]) )
                            {
                                v = max(v,cur->get_bb_outputs()[j]);
                            }
                        }
                        row_Z.set(0,1,v.value());// 2nd column: constraint model
                        break;
                        
                    case NOMAD::SGTELIB_MODEL_FEASIBILITY_UNDEFINED:
                        out << "UNDEFINED";
                        break;
                }// end switch
                
                add_Z.add_rows(row_Z);
                
                if ( cur->is_feasible(0.) )
                {
                    if ( display_update && (!_found_feasible) ) out << " (feasible!)";
                    _found_feasible = true;
                }
                
                
            }// end if valid_point
            next_highest_tag = std::max(next_highest_tag,tag);
            
            if ( display_update )
                out << std::endl;

        }// end if tag
        cur = cache.next();
    }// end boucle cache
    
    _highest_tag = next_highest_tag;
    
    if ( display_update )
    {
        out << "next_highest_tag: " << next_highest_tag << std::endl;
        out << "Current nb of points: " << _trainingset->get_nb_points() << std::endl;
    }
    
    
    if ( add_X.get_nb_rows() > 0 )
    {
        // Build the model
        if ( display_update )
            out << "Add points...";

        _trainingset->add_points( add_X , add_Z );

        if ( display_update )
            out << " OK" << std::endl << "Build_model...";

        _model->build();
        
        if ( display_update )
            out << " OK." << std::endl;
    }
    
    // Check if the model is ready
    _ready = _model->is_ready();
    
    if ( display_update )
    {
        out << "New nb of points: " << _trainingset->get_nb_points() << std::endl;
        out << "Ready: " << _ready << std::endl;
    }
    
    // Update the bounds of the model
    _set_model_bounds(&add_X);
    
    if ( display_update )
        out.close_block();
    
}// Fin de la methode



/*------------------------------------------------------------------------*/
/*       During the update of the sgtelib_model model,                    */
/*           update the bounds of the model                               */
/*------------------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::_set_model_bounds (SGTELIB::Matrix * X)
{
    if ( _p.get_dimension() != X->get_nb_cols() )
    {
        throw NOMAD::Exception ( __FILE__ , __LINE__ ,
                                "Sgtelib_Model_Manager::_set_model_bounds() dimension does not match" );
    }
    
    int nb_dim = X->get_nb_cols();
    int nb_points = X->get_nb_rows();
    
    // Build model bounds
    NOMAD::Double lb;
    NOMAD::Double ub;
    
    for ( int j=0 ; j<nb_dim ; j++)
    {
        lb = _model_lb.get_coord(j);
        ub = _model_ub.get_coord(j);
        for ( int p=0 ; p<nb_points ; p++ )
        {
            lb = min( lb , NOMAD::Double(X->get(p,j)) );
            ub = max( ub , NOMAD::Double(X->get(p,j)) );
        }
        _model_lb.set_coord(j,lb);
        _model_ub.set_coord(j,ub);
    }
}



/*------------------------------------------------------------------------*/
/*                          Extended Bounds                               */
/*------------------------------------------------------------------------*/

NOMAD::Point NOMAD::Sgtelib_Model_Manager::get_extended_lb(void)
{
    NOMAD::Point ext_lb = _p.get_lower_bound();
    NOMAD::Double vi;
    for ( int i=0 ; i < _p.get_dimension() ; i++ )
    {
        vi = _p.get_lower_bound().get_coord(i);
        if ( ( ! vi.is_defined() ) || ( isnan(vi.value() ) ) )
            ext_lb[i] = _model_lb[i] - max(Double(10.0),_model_ub[i]-_model_lb[i]);
    }
    return ext_lb;
}//

NOMAD::Point NOMAD::Sgtelib_Model_Manager::get_extended_ub(void)
{
    NOMAD::Point ext_ub = _p.get_upper_bound();
    NOMAD::Double vi;
    for ( int i = 0 ; i < _p.get_dimension() ; i++ )
    {
        vi = _p.get_upper_bound().get_coord(i);
        if ( (!vi.is_defined()) || (isnan(vi.value())) )
            ext_ub[i] = _model_ub[i] + max(Double(10.0),_model_ub[i]-_model_lb[i]);
    }
    return ext_ub;
}//


/*----------------------------------------------------------------*/
/*     compute model h and f values given one blackbox output     */
/*----------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::eval_h  ( const NOMAD::Point  & bbo    ,
                                            NOMAD::Double       & h       ) const
{
    const NOMAD::Double h_min = _p.get_h_min();
    const NOMAD::hnorm_type h_norm =_p.get_h_norm();
    
    h = 0.0;
    const int m = bbo.size();
    const std::vector<NOMAD::bb_output_type> bbot = _p.get_bb_output_type();
    
    if ( m != static_cast<int>(bbot.size()) )
    {
        std::cout << "Sgtelib_Model_Manager::eval_h() called with an invalid bbo argument" << std::endl;
        throw NOMAD::Exception ( __FILE__ , __LINE__ ,
                                "Sgtelib_Model_Manager::eval_h() called with an invalid bbo argument" );
    }
    NOMAD::Double bboi;
    for ( int i = 0 ; i < m ; ++i )
    {
        bboi = bbo[i];
        if ( bboi.is_defined() )
        {
            if ( bbot[i] == NOMAD::EB || bbot[i] == NOMAD::PEB_E )
            {
                if ( bboi > h_min )
                {
                    h = +INF;
                    return;
                }
            }
            else if ( ( bbot[i] == NOMAD::FILTER ||
                       bbot[i] == NOMAD::PB     ||
                       bbot[i] == NOMAD::PEB_P     ) )
            {
                if ( bboi > h_min )
                {
                    switch ( h_norm )
                    {
                        case NOMAD::L1:
                            h += bboi;
                            break;
                        case NOMAD::L2:
                            h += bboi * bboi;
                            break;
                        case NOMAD::LINF:
                            if ( bboi > h )
                                h = bboi;
                            break;
                    }
                }
            }
            
        }
    }
    if ( h_norm == NOMAD::L2 )
        h = h.sqrt();
    
}// Fin de la methode



/*------------------------------------------------------------------------*/
/*          Compute which formulation must be used in the eval_x          */
/*------------------------------------------------------------------------*/
const NOMAD::sgtelib_model_formulation_type NOMAD::Sgtelib_Model_Manager::get_formulation( void )
{
    
    NOMAD::sgtelib_model_formulation_type formulation = _p.get_SGTELIB_MODEL_FORMULATION();
    //std::cout << __FILE__ << __LINE__ << " ready = " << _ready ;
    if ( (formulation != NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN) && ( ! _ready) ){
        formulation = NOMAD::SGTELIB_MODEL_FORMULATION_D;
      //std::cout << " ... use formulation D";
    }
    //std::cout << std::endl;
    
    return formulation;
}

/*------------------------------------------------------------------------*/
/*          Check that h & f are defined, and if not, correct it          */
/*------------------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Manager::check_hf ( NOMAD::Eval_Point   * x )
{
    
    NOMAD::Double f = x->get_f();
    NOMAD::Double h = x->get_h();
    
    if ( ! f.is_defined() )
        f = x->get_bb_outputs().get_coord(_p.get_index_obj().front());
    
    if ( ! h.is_defined() )
        eval_h ( x->get_bb_outputs() , h );
    
    if ( ( ! f.is_defined()) || ( ! h.is_defined()) )
    {
        f = INF;
        h = INF;
    }
    x->set_f(f);
    x->set_h(h);
    
}


/*------------------------------------------------------------------------*/
/*                evaluate the sgtelib_model model at a given point       */
/*------------------------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Manager::eval_x ( NOMAD::Eval_Point   * x          ,
	                                          const NOMAD::Double & h_max      ,
                                          	bool                & count_eval )
{
    int i;
    const int dim = _p.get_dimension();
    const int nbbo = _p.get_bb_nb_outputs();
    const NOMAD::Double diversification = _p.get_SGTELIB_MODEL_DIVERSIFICATION();
    const NOMAD::Display & out = _p.out();    

    const bool bool_display = (string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"X"));
    
    if ( bool_display )
    {
        out.open_block("Model evaluation");
        out << "X = (";
        for ( i = 0 ; i < dim ; i++ )
        {
            std::cout << x->get_coord(i).value() << " ";
        }
        out << ")" << std::endl;
    }
    
    
    // --------------------- //
    // In/Out Initialisation //
    // --------------------- //
    
    // Creation of matrix for input / output of SGTELIB model
    SGTELIB::Matrix X_predict   ( "X_predict"   , 1 , dim );
    
    // Set the input matrix
    for ( i = 0 ; i < dim ; i++ )
    {
        X_predict.set(0,i, x->get_coord(i).value()   );
    }
    // reset point outputs:
    // Par défaut, on met tout à -1
    for ( i = 0 ; i < nbbo ; ++i )
        x->set_bb_output ( i , NOMAD::Double(-1) );
    
    
    // ------------------------- //
    //   Objective Prediction    //
    // ------------------------- //
    
    
    // Declaration of the stastistical measurements
    NOMAD::Double pf = 1; // P[x]
    NOMAD::Double f = 0; // predicted mean of the objective
    NOMAD::Double sigma_f = 0; // predicted variance of the objective
    NOMAD::Double pi = 0; // probability of improvement
    NOMAD::Double ei = 0; // expected improvement
    NOMAD::Double efi = 0; // expected feasible improvement
    NOMAD::Double pfi = 0; // probability of feasible improvement
    NOMAD::Double mu = 0; // uncertainty on the feasibility
    NOMAD::Double penalty = 0; // exclusion area penalty
    NOMAD::Double d = 0; // Distance to closest point of the cache
    NOMAD::Double h = 0; // Constraint violation
    
    // FORMULATION USED IN THIS EVAL_X
    const NOMAD::sgtelib_model_formulation_type formulation = get_formulation();
    
    
    // Shall we compute statistical criterias
    bool use_statistical_criteria = false;
    switch (formulation)
    {
        case NOMAD::SGTELIB_MODEL_FORMULATION_FS:
            use_statistical_criteria = (diversification != 0);
            break;
        case NOMAD::SGTELIB_MODEL_FORMULATION_FSP:
        case NOMAD::SGTELIB_MODEL_FORMULATION_EIS:
        case NOMAD::SGTELIB_MODEL_FORMULATION_EFI:
        case NOMAD::SGTELIB_MODEL_FORMULATION_EFIS:
        case NOMAD::SGTELIB_MODEL_FORMULATION_EFIM:
        case NOMAD::SGTELIB_MODEL_FORMULATION_EFIC:
        case NOMAD::SGTELIB_MODEL_FORMULATION_PFI:
            use_statistical_criteria = true;
            break;
        case NOMAD::SGTELIB_MODEL_FORMULATION_D:
            use_statistical_criteria = false;
            break;
        case NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN:
        case NOMAD::SGTELIB_MODEL_FORMULATION_UNDEFINED:
            throw SGTELIB::Exception ( __FILE__ , __LINE__ , "Forbiden formulation" );
            break;
    }
    if ( bool_display )
    {
        out << "Formulation: "
        << sgtelib_model_formulation_type_to_string ( formulation )
        << "; compute stat: " << use_statistical_criteria
        << "; found_feasible : " << _found_feasible << std::endl;
    }
    
    
    // Init the matrices for prediction
    SGTELIB::Matrix   M_predict (   "M_predict" , 1 , _nb_models );
    SGTELIB::Matrix STD_predict ( "STD_predict" , 1 , _nb_models );
    SGTELIB::Matrix CDF_predict ( "CDF_predict" , 1 , _nb_models );
    SGTELIB::Matrix  EI_predict (  "EI_predict" , 1 , _nb_models );
    
    // Prediction
    if ( formulation == NOMAD::SGTELIB_MODEL_FORMULATION_D )
    {
        d = _trainingset->get_distance_to_closest(X_predict).get(0,0);
        if ( bool_display )
        {
            out << "d = " << d << std::endl;
        }
    }
    else if ( formulation == NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
    {
        throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                                  "Sgtelib_Model_Manager::eval_x: Formulation Extern should not been called in this context." );
    }
    else
    {
        if ( bool_display )
        {
            out << "Predict... ";
        }
        _model->build();
        _model->check_ready(__FILE__,__FUNCTION__,__LINE__);
        if (use_statistical_criteria)
            _model->predict( X_predict , &M_predict , &STD_predict , &EI_predict , &CDF_predict );
        else
            _model->predict(X_predict,&M_predict);
        
        if ( bool_display )
            out << "ok" << std::endl;
    }
    
    // Get the prediction from the matrices
    f = M_predict.get(0,0);
    
    if ( use_statistical_criteria )
    {
        // If no feasible points is found so far, then sigma_f, ei and pi are bypassed.
        if ( _found_feasible )
        {
            sigma_f = STD_predict.get(0,0);
            pi      = CDF_predict.get(0,0);
            ei      = EI_predict.get(0,0);
        }
        else
        {
            sigma_f = 1.0; // This inhibits the exploration term in regard to the objective
            pi      = 1.0; // This implies that pfi = pf
            ei      = 1.0; // This implies that efi = pf
        }
        if ( bool_display )
            out << "F = " << f << " +/- " << sigma_f << std::endl;
        
    }
    else
    {
        if ( bool_display )
            out << "F = " << f << std::endl;
    }
    
    
    // ====================================== //
    // Constraints display                    //
    // ====================================== //
    
    if ( bool_display )
    {
        switch ( _p.get_SGTELIB_MODEL_FEASIBILITY() )
        {
            case NOMAD::SGTELIB_MODEL_FEASIBILITY_C:
                if (use_statistical_criteria){
                  for ( i = 1 ; i < _nb_models ; i++ )
                  {
                      out << "C" << i << " = " << M_predict.get(0,i)
                      << " +/- "    << STD_predict.get(0,i)
                      << " (CDF : " << CDF_predict.get(0,i) << ")" << std::endl;
                  }
                }
                else{
                  out << "C = [ ";
                  for ( i = 1 ; i < _nb_models ; i++ ) out << M_predict.get(0,i) << " ";
                  out << " ]" << std::endl;
                }
                break;
            case NOMAD::SGTELIB_MODEL_FEASIBILITY_H:
                out << "Feasibility_Method : H (Aggregate prediction)" << std::endl;
                out << "H = " << M_predict.get(0,1) << " +/- " << STD_predict.get(0,1) << " (CDF : " << CDF_predict.get(0,1) << ")" << std::endl;
                break;
            case NOMAD::SGTELIB_MODEL_FEASIBILITY_B:
                out << "Feasibility_Method : B (binary prediction)" << std::endl;
                out << "B = " << M_predict.get(0,1)  << " (CDF : " << CDF_predict.get(0,1) << ")" << std::endl;
                break;
            case NOMAD::SGTELIB_MODEL_FEASIBILITY_M:
                out << "Feasibility_Method : M (Biggest constraint prediction)" << std::endl;
                out << "M = " << M_predict.get(0,1) << " +/- " << STD_predict.get(0,1) << " (CDF : " << CDF_predict.get(0,1) << ")" << std::endl;
                break;
            case NOMAD::SGTELIB_MODEL_FEASIBILITY_UNDEFINED:
                out << "SGTELIB_MODEL_FEASIBILITY_UNDEFINED" << std::endl;
                break;
        }
    }
    
    
    
    
    // ====================================== //
    // Computation of statistical criteria    //
    // ====================================== //
    if ( use_statistical_criteria )
    {
        pf = 1; // General probability of feasibility
        NOMAD::Double pfj; // Probability of feasibility for constrait cj
        NOMAD::Double L2 = 0;
        if ( _p.has_constraints() )
        {
            // Use the CDF of each output in C
            // If there is only one output in C (models B, H and M) then pf = CDF)
            for ( i = 1 ; i < _nb_models ; i++ )
            {
                pfj = CDF_predict.get(0,i);
                L2 += max( 0 , M_predict.get(0,i)).pow2();
                pf *= pfj;
            }
        } // end (if constraints)
        if ( ( !_found_feasible ) && (pf == 0) )
        {
            pf = 1.0/(1.0+L2);
            if ( bool_display )
            {
                out << "pf = 0 and L2 = " << L2 << " => pF = " << pf << std::endl;
            }
        }
        pfi = pi*pf;
        efi = ei*pf;
        mu = 4*pf*(1-pf);
    }
    
    // ====================================== //
    // Application of the formulation         //
    // ====================================== //
    const std::vector<NOMAD::bb_output_type> bbot = _p.get_bb_output_type();
    NOMAD::Double obj;
    int obj_index = _p.get_index_obj().front();
    int k;
    switch ( formulation )
    {
      case NOMAD::SGTELIB_MODEL_FORMULATION_FS:
          // Define obj
          obj = f - diversification*sigma_f;
          // Set constraints
          k = 0;
          for ( i = 0 ; i < nbbo ; i++ )
          {
            if ( bbot[i] != NOMAD::OBJ )
            {
                x->set_bb_output( i , M_predict.get(0,k+1) - diversification*STD_predict.get(0,k+1) );
                k++;
            }
          }
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_FSP:
          // Define obj
          obj = f - diversification*sigma_f;
          // Set constraints
          for ( i = 0 ; i < nbbo ; i++ )
          {
              if ( bbot[i] != NOMAD::OBJ )
                  x->set_bb_output( i , 0.5 - pf );
          }
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EIS:
          // Define obj
          obj = - ei - diversification*sigma_f;
          // Set constraints
          k = 0;
          for ( i = 0 ; i < nbbo ; i++ )
          {
              if ( bbot[i] != NOMAD::OBJ )
              {
                  x->set_bb_output( i , M_predict.get(0,k+1) - diversification*STD_predict.get(0,k+1) );
                  k++;
              }
          }
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EFI:
          obj = -efi;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EFIS:
          obj = -efi - diversification*sigma_f;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EFIM:
          obj = -efi - diversification*sigma_f*mu;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EFIC:
          obj = -efi - diversification*( ei*mu + pf*sigma_f);
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_PFI:
          obj = -pfi;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_D:
          obj = -d;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN:
          out<< "SGTELIB_MODEL_FORMULATION_EXTERN" << std::endl;
          break;
      case NOMAD::SGTELIB_MODEL_FORMULATION_UNDEFINED:
          out<< "SGTELIB_MODEL_FORMULATION_UNDEFINED" << std::endl;
          break;
    }
    
    // ------------------------- //
    //   exclusion area          //
    // ------------------------- //
    
    double tc = _p.get_SGTELIB_MODEL_EXCLUSION_AREA().value();
    if ( tc > 0.0 )
    {
        penalty = _model->get_exclusion_area_penalty ( X_predict,tc ).get(0,0);
        obj += penalty;
    }
    
    
    
    // ------------------------- //
    //   Set obj                 //
    // ------------------------- //
    x->set_bb_output( obj_index , obj );
    eval_h ( x->get_bb_outputs() , h );
    x->set_f ( obj );
    x->set_h ( h );
    
    // ================== //
    //       DISPLAY      //
    // ================== //
    if ( bool_display )
    {
        if ( use_statistical_criteria )
        {
            out << "f_min                    f_min = " << _trainingset->get_f_min() << std::endl;
            out << "Probability of Feasibility PF  = " << pf << std::endl;
            out << "Feasibility Uncertainty    mu  = " << mu << std::endl;
            out << "Probability Improvement    PI  = " << pi << std::endl;
            out << "Exptected Improvement      EI  = " << ei << std::endl;
            out << "Proba. of Feasible Imp.    PFI = " << pfi << std::endl;
            out << "Expected Feasible Imp.     EFI = " << efi << std::endl;
        }
        out << "Exclusion area penalty = " << penalty << std::endl;
        out << "Model Output = (" << x->get_bb_outputs() << ")" << std::endl;
        if ( isnan( pf.value() ) || isnan( pi.value() ) )
        {
            throw SGTELIB::Exception ( __FILE__ , __LINE__ ,
                                      "Sgtelib_Model_Manager::eval_x: nan values in pi or pf." );
        }
        out.close_block();
    }
    
    
    // ================== //
    // Statut de sortie   //
    // ================== //
    count_eval = true;
    x->set_eval_status ( NOMAD::EVAL_OK );
    return true;
    
    
}// Fin de la methode


/*------------------------------------------------------------------------*/
/*         get fmin from the training set                                 */
/*------------------------------------------------------------------------*/
NOMAD::Double NOMAD::Sgtelib_Model_Manager::get_f_min (void)
{
    if ( _trainingset->is_ready() )
    {
        std::cout << "(get_f_min : is ready!, " << _trainingset->get_nb_points() << ")" << std::endl;
        return _trainingset->get_f_min();
    }
    else
    {
        std::cout << "(get_f_min : NOT ready!)" << std::endl;
        return NaN;
    }
}//

