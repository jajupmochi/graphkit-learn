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
 \file   Sgtelib_Model_Search.cpp
 \brief  Model search using the sgtelib_model surrogate library.
 \author Bastien Talgorn
 \date   2013-04-25
 \see    Sgtelib_Model_Manager.cpp
 */

#include "Sgtelib_Model_Search.hpp"

/*-----------------------------------*/
/*           reset (virtual)         */
/*-----------------------------------*/
void NOMAD::Sgtelib_Model_Search::reset ( void )
{
    delete _start_point_1;
    delete _start_point_2;
}

/*--------------------------------------------------------*/
/*  delete a list of points: one version for points, and  */
/*  one version for evaluation points (static, private)   */
/*--------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Search::clear_pts ( std::vector<NOMAD::Point *> & pts )
{
    size_t k , n = pts.size();
    for ( k = 0 ; k < n ; ++k )
        delete pts[k];
    pts.clear();
}//

void NOMAD::Sgtelib_Model_Search::clear_pts ( std::vector<NOMAD::Eval_Point *> & pts )
{
    size_t k , n = pts.size();
    for ( k = 0 ; k < n ; ++k )
        delete pts[k];
    pts.clear();
}//


/*------------------------------------------------------------------*/
/*                             the search                           */
/*------------------------------------------------------------------*/
/* Search parameters:                                               */
/* ------------------                                               */
/*                                                                  */
/*  . MODEL_SEARCH: flag to activate the model search (MS)          */
/*                  (here its value is NOMAD::Sgtelib_Model)        */
/*                                                                  */
/*  . MODEL_SEARCH_OPTIMISTIC: if true, the direction from the      */
/*                             model center to the trial point      */
/*                             is computed and possibly  used       */
/*                             in the speculative search            */
/*                             default=yes                          */
/*                                                                  */
/*  . MODEL_SEARCH_PROJ_TO_MESH: project or not to mesh             */
/*                               default=yes                        */
/*                                                                  */
/*  . MODEL_SEARCH_MAX_TRIAL_PTS: limit on the number of trial      */
/*                                points for one search             */
/*                                default=10                        */
/*                                                                  */
/*------------------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Search::search ( NOMAD::Mads              & mads           ,
                                          int                      & nb_search_pts  ,
                                          bool                     & stop           ,
                                          NOMAD::stop_type         & stop_reason    ,
                                          NOMAD::success_type      & success        ,
                                          bool                     & count_search   ,
                                          const NOMAD::Eval_Point *& new_feas_inc   ,
                                          const NOMAD::Eval_Point *& new_infeas_inc   )
{
    new_feas_inc  = new_infeas_inc = NULL;
    nb_search_pts = 0;
    success       = NOMAD::UNSUCCESSFUL;
    count_search  = false;
    const NOMAD::Display    & out = _p.out();
    const NOMAD::dd_type display_degree = out.get_search_dd();
    //const bool bool_display = ( string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"S") or (display_degree == NOMAD::FULL_DISPLAY) );
    const bool bool_display = ( _p.get_SGTELIB_MODEL_DISPLAY().size() || (display_degree == NOMAD::FULL_DISPLAY) );
    
    // initial displays:
    if ( bool_display )
    {
        std::ostringstream oss;
        oss << NOMAD::MODEL_SEARCH << " #"
        << _all_searches_stats.get_MS_nb_searches();
        out << std::endl;
        out.open_block ( oss.str() );
    }
    
    
    
    NOMAD::Double fmin_old;
    if (mads.get_best_feasible())
        fmin_old = mads.get_best_feasible()->get_f();
    else
        fmin_old = NOMAD::INF;
    
    _one_search_stats.reset();
    
    
    int display_lim = 15;
    
    if ( stop )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): not performed (stop flag is active)"
            << std::endl;
        return;
    }
    
    // active cache (we accept only true function evaluations):
    const NOMAD::Cache & cache = mads.get_cache();
    if ( cache.get_eval_type() != NOMAD::TRUTH )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): not performed on surrogates"
            << std::endl;
        return;
    }
    
    // check that there is one objective exactly:
    const std::list<int> & index_obj_list = _p.get_index_obj();
    if ( index_obj_list.empty() )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): not performed with no objective function"
            << std::endl;
        return;
    }
    if ( index_obj_list.size() > 1 )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): not performed with biobjective"
            << std::endl;
        return;
    }
    
    // active barrier:
    const NOMAD::Barrier & barrier = mads.get_true_barrier();
    
    // get the incumbent:
    const NOMAD::Eval_Point * incumbent = barrier.get_best_feasible();
    if ( !incumbent )
        incumbent = barrier.get_best_infeasible();
    if ( !incumbent )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): no incumbent"
            << std::endl;
        return;
    }
    
    // get and check the signature, and compute the dimension:
    NOMAD::Signature * signature = incumbent->get_signature();
    
    if ( !signature )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): no signature"
            << std::endl;
        return;
    }
    
    int n = signature->get_n();
    
    if ( n != incumbent->size() )
    {
        if ( bool_display )
            out << "Sgtelib_Model_Search::search(): incompatible signature"
            << std::endl;
        return;
    }
    
    
    
    // from this point the search is counted:
    count_search = true;
    _one_search_stats.add_MS_nb_searches();
    
    // mesh:
    NOMAD::Point delta_m;
    signature->get_mesh()->get_delta(delta_m);
    
    // initial displays:
    if ( bool_display )
    {
        out << "number of cache points: " << cache.size()    << std::endl;
        out << "mesh size parameter: ( " << delta_m << " )" << std::endl;
        out << "incumbent: ( ";
        incumbent->NOMAD::Point::display( out , " " , 2 , NOMAD::Point::get_display_limit() );
        out << " )" << std::endl;
    }
    
    // Get mads & ev_ctrl
    NOMAD::Stats & stats = mads.get_stats();
    NOMAD::Evaluator_Control * ev_control = _sgtelib_model_manager->get_evaluator_control();
    
    const int kkmax = _p.get_SGTELIB_MODEL_TRIALS();
    
    // MULTIPLE_SEARCH
    for (int kk=0 ; kk<kkmax ; kk++)
    {
        
        /*----------------*/
        /*  Model update  */
        /*----------------*/
        _sgtelib_model_manager->update();
        
        /*----------------*/
        /*  oracle points */
        /*----------------*/
        
        std::vector<NOMAD::Point *> oracle_pts;
        if ( ! create_oracle_pts ( mads          ,
                                  *incumbent     ,
                                  delta_m        ,
                                  out            ,
                                  display_degree ,
                                  display_lim    ,
                                  oracle_pts     ,
                                  stop           ,
                                  stop_reason      ) && stop )
        {
            
            if ( bool_display )
                out << "fail: create_oracle_pts" << std::endl;
            
            
            // delete oracle_pts:
            NOMAD::Sgtelib_Model_Search::clear_pts ( oracle_pts );
            
            // quit:
            if ( bool_display )
                out << NOMAD::close_block ( "algorithm stop" ) << std::endl;
            
            return;
        }
        
        /*------------------*/
        /*  Register points */
        /*------------------*/
        
        // add the trial points to the evaluator control for evaluation:
        int i , nop = static_cast<int>(oracle_pts.size());
        
        for ( i = 0 ; i < nop ; ++i )
            register_point ( *oracle_pts[i] ,
                            *signature     ,
                            *incumbent     ,
                            display_degree );
        
        // display the evaluator control list of points:
        if ( bool_display )
        {
            out.open_block ( "list of trial points" );
            const std::set<NOMAD::Priority_Eval_Point> & lop = ev_control->get_eval_lop();
            std::set<NOMAD::Priority_Eval_Point>::const_iterator it , end = lop.end();
            nop = static_cast<int>(lop.size());
            for ( it = lop.begin() , i = 0 ; it != end ; ++it , ++i )
            {
                out << "#";
                out.display_int_w ( i , nop );
                out << " x=( ";
                it->get_point()->NOMAD::Point::display ( out , " " , 15 , -1 );
                out << " )" << std::endl;
            }
            out.close_block();
        }
        
        // delete the trial points
        NOMAD::Sgtelib_Model_Search::clear_pts ( oracle_pts   );
        
        nb_search_pts = ev_control->get_nb_eval_points();
        
        
        /*---------------------------*/
        /* Evaluate the trial points */
        /*---------------------------*/
        bool cache_hit = false;
        if ( nb_search_pts == 0 )
        {
            if ( bool_display )
                out << std::endl << "no trial point" << std::endl;
        }
        else
        {
            _one_search_stats.update_MS_max_search_pts ( nb_search_pts );
            int bbe        = stats.get_bb_eval();
            int sgte_eval  = stats.get_sgte_eval ();
            int cache_hits = stats.get_cache_hits();
            
            new_feas_inc = new_infeas_inc = NULL;
            
            ev_control->disable_model_eval_sort();
            
            std::list<const NOMAD::Eval_Point *> * evaluated_pts = NULL;
            if ( bool_display )
                evaluated_pts = new std::list<const NOMAD::Eval_Point *>;
            
            ev_control->eval_list_of_points ( _type                  ,
                                             mads.get_true_barrier() ,
                                             mads.get_sgte_barrier() ,
                                             mads.get_pareto_front() ,
                                             stop                    ,
                                             stop_reason             ,
                                             new_feas_inc            ,
                                             new_infeas_inc          ,
                                             success                 ,
                                             evaluated_pts             );
            ev_control->enable_model_eval_sort();
        
            delete evaluated_pts;
            
            cache_hit = (cache_hits<stats.get_cache_hits());
            
            // update stats:
            _one_search_stats.add_MS_bb_eval    ( stats.get_bb_eval   () - bbe        );
            _one_search_stats.add_MS_sgte_eval  ( stats.get_sgte_eval () - sgte_eval  );
            _one_search_stats.add_MS_cache_hits ( stats.get_cache_hits() - cache_hits );
            _one_search_stats.add_MS_pts ( nb_search_pts );
            
            // Success ??
            if ( success >= NOMAD::FULL_SUCCESS )
            {
                success = NOMAD::FULL_SUCCESS_STAY;
                _one_search_stats.add_MS_success();
                break;
            }
        }
        if ( bool_display )
        {
            out << "Trial outcome: ";
            if ( cache_hit )
                out << "=> Cache hit" << std::endl;
            else if ( nb_search_pts == 0 )
                out << "=> No trial point" << std::endl;
            else
                out << "=> " << success << std::endl;
        }
        
    } // end loop kk
    
    // update stats objects:
    stats.update_model_stats   ( _one_search_stats );
    _all_searches_stats.update ( _one_search_stats );
    
    // final display:
    if ( bool_display )
    {
        const NOMAD::Eval_Point * bf = mads.get_best_feasible();
        
        out << "Sgtelib model search: " << success << std::endl;
        if ( ( success >= NOMAD::FULL_SUCCESS ) && (bf) )
        {
            const double fmin_new = bf->get_f().value();
            out << "fmin (old-->new): " << fmin_old << " --> " << fmin_new << std::endl;
            out << "Improvement: " << fmin_old-fmin_new << std::endl;
        }
        // Close block
        std::ostringstream oss;
        oss << "end of " << NOMAD::MODEL_SEARCH << " (" << success << ")";
        out << NOMAD::close_block ( oss.str() ) << std::endl;
    }
    
}// end search







/*-------------------------------------------------------------------*/
/*  create a list of oracle points, given by the model optimization  */
/*  (private)                                                        */
/*-------------------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Search::create_oracle_pts
( const NOMAD::Mads                     & mads          ,
 const NOMAD::Point                     & incumbent      ,
 const NOMAD::Point                     & delta_m        ,
 const NOMAD::Display                   & out            ,
 NOMAD::dd_type                           display_degree ,
 int                                      display_lim    ,
 std::vector<NOMAD::Point *>            & oracle_pts     ,
 bool                                   & stop           ,
 NOMAD::stop_type                       & stop_reason      )
{
    int i;
    const NOMAD::Cache & cache = mads.get_cache();
    const bool bool_display = (display_degree == NOMAD::FULL_DISPLAY) ;
    
    
    // starting points selection:
    //---------------------------
    const NOMAD::Eval_Point * x0s[4];
    x0s[0] = x0s[1] = x0s[2] = x0s[3] = NULL;
    
    i = 0;
    if ( mads.get_best_feasible() )
    {
        x0s[i] = new NOMAD::Eval_Point( *mads.get_best_feasible() );
        i++;
    }
    if ( mads.get_best_infeasible() )
    {
        x0s[i] = new NOMAD::Eval_Point( *mads.get_best_infeasible() );
        i++;
    }
    if ( _start_point_1 )
    {
        const int m = _p.get_bb_nb_outputs();
        x0s[i] = new NOMAD::Eval_Point( *_start_point_1 , m );
        i++;
    }
    if ( _start_point_2 )
    {
        const int m = _p.get_bb_nb_outputs();
        x0s[i] = new NOMAD::Eval_Point( *_start_point_2 , m );
        i++;
    }
    
    if ( bool_display ) out.open_block ( "oracle points construction" );
    
    // reset oracle points:
    NOMAD::Sgtelib_Model_Search::clear_pts ( oracle_pts );
    NOMAD::Double         f_model , h_model;
    
    if ( !x0s[0] && !x0s[1] && !x0s[2] && !x0s[3] )
    {
        if ( bool_display )
            out << std::endl
            << NOMAD::close_block ( "oracle points error: no model starting point" )
            << std::endl;
        return false;
    }
    
    // optimize model:
    // ---------------
    NOMAD::Clock clock;
    
    // CALLERXXX
    bool optimization_ok = optimize_model ( cache          ,
                                           incumbent      ,
                                           delta_m        ,
                                           x0s            ,
                                           out            ,
                                           display_degree ,
                                           oracle_pts     ,
                                           stop           ,
                                           stop_reason    );
    
    
    
    
    _one_search_stats.add_optimization_time ( clock.get_CPU_time() );
    
    
    // Check for errors:
    // -----------------
    if ( stop || ! optimization_ok || oracle_pts.size()==0 )
    {
        std::string error_str;
        if ( oracle_pts.size()==0 )
            error_str = "no model optimization solution";
        else
        {
            error_str = ( stop ) ? "algorithm stop" : "model optimization error";
        }
        if ( bool_display )
            out << std::endl << NOMAD::close_block ( "oracle points error: " + error_str ) << std::endl;
        return false;
    }
    
    
    // Display:
    //---------
    if ( bool_display )
    {
        out << std::endl
        << NOMAD::close_block ( "end of oracle points construction" )
        << std::endl;
    }
    
    // Delete x0s
    for ( i = 0 ; i < 4 ; i++)
        if ( x0s[i] )
            delete x0s[i];
    
    return true;
}// end create_oracle_points


/*------------------------------------------------------*/
/*  filter cache                                        */
/*------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Search::filter_cache (  const NOMAD::Display & out            ,
                                                const NOMAD::Cache   & cache          ,
                                                const NOMAD::Cache   & cache_surrogate,
                                                const int            & nb_candidates,
                                                const double         & delta_m_norm,
                                                std::vector<NOMAD::Point *> & oracle_pts )
{
    const bool bool_display = (string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"F"));
    const int pp = cache_surrogate.size();
    const int n = _p.get_dimension();
    int i,j,k;
    double d,v;
    const NOMAD::sgtelib_model_formulation_type formulation = _sgtelib_model_manager->get_formulation();
    
    //===========================================
    // Compute objective, aggregate constraint,
    // distance to main cache, and copy the search cache.
    // ==========================================
    // Objective function (prediction)
    double * f = new double [pp];
    // Aggregate constraint (prediction)
    double * h = new double [pp];
    // Feasibility value (max of cj)
    double * hmax = new double [pp];
    // Distance to main cache.
    double * DX = new double [pp];
    // Copy of the search point position.
    double ** S = new double * [pp];
    
    const NOMAD::Eval_Point * cur1;
    const NOMAD::Eval_Point * cur2;
    NOMAD::Point bbo;
    const std::vector<NOMAD::bb_output_type> bbot = _p.get_bb_output_type();
    const int nbbo = _p.get_bb_nb_outputs();
    
    cur1 = cache_surrogate.begin();
    i = 0;
    
    while ( cur1 )
    {
        NOMAD::Eval_Point * x = new NOMAD::Eval_Point(*cur1);
        _sgtelib_model_manager->check_hf(x);
        f[i] = x->get_f().value();
        h[i] = x->get_h().value();
        
        if ( x->check_nan() )
        {
            f[i] = +INF;
            h[i] = +INF;
        }
        
        // hmax(x) = max_j c_j(x)
        hmax[i] = -INF;
        bbo = x->get_bb_outputs();
        for ( j=0 ; j<nbbo ; j++)
        {
            if ( bbot_is_constraint(bbot[j]) )
                hmax[i] = std::max(hmax[i],bbo[j].value());
        }
        
        
        // Compute distance to main cache
        d = +INF;
        cur2 = cache.begin();
        while ( cur2 )
        {
            d = std::min(d,((*cur1)-(*cur2)).squared_norm().value());
            cur2 = cache.next();
        }
        DX[i] = sqrt(d);
        // Copy coordinates in S
        S[i] = new double [n];
        for (j=0 ; j<n ; j++) S[i][j] = cur1->get_coord(j).value();
        // Move to next point
        cur1 = cache_surrogate.next();
        i++;
    }
    
    
    
    //=======================================================
    // Compute the distance between each pair of points of S
    //=======================================================
    if ( bool_display ){
        out << "Compute distances" << std::endl;
    }
    // Allocate DSS
    double ** DSS = new double * [pp];
    for (i=0 ; i<pp ; i++) DSS[i] = new double [pp];
    // Compute the distance between each pair of points
    for (i=0 ; i<pp ; i++)
    {
        DSS[i][i] = 0;
        for (j=i+1 ; j<pp ; j++)
        {
            d = 0;
            for (k=0 ; k<n ; k++)
            {
                v = S[i][k]-S[j][k];
                d += v*v;
            }
            DSS[i][j] = sqrt(d);
            DSS[j][i] = DSS[i][j];
        }
    }
    // delete S
    for (i=0 ; i<pp ; i++)
        delete [] S[i];
    delete [] S;
    
    
    
    //=======================================================
    // Compute initial isolation distances
    // The isolation of a point i of the surrogate cache,
    // is the distance to the closest point that is better than i.
    //=======================================================
    if ( bool_display )
    {
        out << "Compute isolations" << std::endl;
    }
    double * d_isolation = new double [pp];
    for (i=0 ; i<pp ; i++)
    {
        d = +INF;
        for (j=0 ; j<pp ; j++)
        {
            // If the point j is better than i
            if ( (h[j]<h[i])||( (h[j]==h[i]) && (f[j]<f[i]) ) )
            {
                d = std::min(d,DSS[i][j]);
            }
        }
        d_isolation[i] = d;
    }
    
    //====================================================
    // Greedy selection
    //===================================================
    
    // Boolean for storing the indexes of selected points
    bool * keep = new bool [pp];
    int nkeep = 0;
    
    // Distance between search points and selected points
    double * DT  = new double [pp];
    double * DTX = new double [pp];
    int * n_isolation  = new int [pp];
    int * n_density = new int [pp];
    // Init these values
    for (i=0 ; i<pp ; i++)
    {
        keep[i] = false;
        DT[i] = +INF;
        DTX[i] = DX[i];
        n_isolation[i] = -1;
        n_density[i] = -1;
    }
    
    
    // Boolean array of size 10 indicating for each method of index 0 to 9,
    // if the method should be used.
    const int NB_METHODMAX = 10;
    bool use_method[NB_METHODMAX];
    int nb_methods = 0;
    for (i=0 ; i<10 ; i++)
    {
        use_method[i] = (string_find(_p.get_SGTELIB_MODEL_FILTER(),itos(i)));
        if (use_method[i])
        {
            nb_methods++;
        }
    }
    
    if ( bool_display )
    {
        out << "Used method: " ;
        
        // Select method
        if ( formulation == NOMAD::SGTELIB_MODEL_FORMULATION_D )
        {
            out << "Method override. Use method 1" << std::endl;
        }
        else
        {
            for (i=0 ; i<10 ; i++)
            {
                if (use_method[i])
                {
                    out << i << " ";
                }
            }
            out << "(total nb methods = " << nb_methods << ")" << std::endl;
        }
    }
    
    if (nb_methods==0)
        throw NOMAD::Exception ( "Sgtelib_Model_Search.cpp" ,
                                 __LINE__ ,"method index non valid" );
    
    
    
    
    
    
    int failure = 0;
    int ni, nmax, iselect;
    double fmin, hmin, dmax;
    int method = 0;
    
    // initial dmin, for method 2
    double dmin = 0;
    // Compute hmax threshold for method 3
    // (Largest strictly negative value of hmax)
    double hmax_threshold = -INF;
    for (i=0 ; i<pp ; i++)
    {
        if (hmax[i]<0)
            hmax_threshold = std::max(hmax_threshold,hmax[i]);
    }
    
    
    
    if ( bool_display )
    {
        out << "Filter: Start greedy selection" << std::endl;
    }
    
    
    while ((nkeep<nb_candidates) && (failure<2*nb_methods))
    {
        
        // Selected point
        iselect = -1;
        
        // Select method
        if ( formulation == NOMAD::SGTELIB_MODEL_FORMULATION_D ){
            method = 1;
        }
        else
        {
            // Otherwise, cycle through all the methods
            while (true)
            {
                method++;
                if (method==NB_METHODMAX)
                    method = 0;
                if (use_method[method])
                    break;
            }
        }
        if ( bool_display )
        {
            out << "Method " << method;
        }
        
        switch (method)
        {
                
                
                
                //----------------------------------------
            case 0:
            {
                // Method 0 : selects the best point
                
                // Select
                fmin = +INF;
                hmin = +INF;
                for (i=0 ; i<pp ; i++)
                {
                    if ((!keep[i]) && (DTX[i]>0))
                    {
                        // Check if i is better than iselect
                        bool bool1 = ( h[i]<hmin );
                        bool bool2 = ( (h[i]==hmin) && (f[i]<fmin) );
                        if ( bool1||bool2 )
                        {
                            hmin = h[i];
                            fmin = f[i];
                            iselect = i;
                        }
                    }
                }
            }
                break;
                
                //----------------------------------------
                // Special case for formulation D
            case 1:
            {
                // Method 1 : selects the most distance point
                dmax = 0;
                for (i=0 ; i<pp ; i++)
                {
                    if ( ( ! keep[i]) && (DTX[i]>=dmax) )
                    {
                        dmax = DTX[i];
                        iselect = i;
                    }
                }
            }
                break;
                
                
                
                //----------------------------------------
            case 2:
            {
                // Method 2, selects the best point but with a minimum distance to points already selected
                
                // Distance threashold
                if ( bool_display )
                {
                    out << ", dmin = " << dmin << std::endl;
                }
                
                // Select
                fmin = +INF;
                hmin = +INF;
                for (i=0 ; i<pp ; i++)
                {
                    if ( ( ! keep[i] ) && ( DTX[i] >= dmin ) )
                    {
                        // Check if i is better than iselect
                        bool bool1 = ( h[i]<hmin );
                        bool bool2 = ( (h[i]==hmin) && (f[i]<fmin) );
                        if ( bool1||bool2 )
                        {
                            hmin = h[i];
                            fmin = f[i];
                            iselect = i;
                        }
                    }
                }
                if ( bool_display && (iselect != -1) )
                {
                    out << "d = " << DTX[iselect] << std::endl;
                    out << "h select = " << hmin << std::endl;
                }
                if (iselect!=-1)
                {
                    dmin =  DTX[iselect] + delta_m_norm;
                }
                
            }
                break;
                //----------------------------------------
            case 3:
            {
                // Select the best points with a good enough value of hmax
                
                // Select
                fmin = +INF;
                for (i=0 ; i<pp ; i++)
                {
                    if ( (!keep[i]) && (hmax[i]<=hmax_threshold) && (f[i]<fmin) && (DTX[i]>delta_m_norm) )
                    {
                        fmin = f[i];
                        iselect = i;
                    }
                }
                if ( iselect == -1)
                    hmax_threshold *= 0.5;
                else
                    hmax_threshold = 2.0*hmax[iselect];
            }
                break;
                //----------------------------------------
            case 4:
            {
                // Select point with highest isolation number
                
                nmax = 0;
                for (i=0 ; i<pp ; i++)
                {
                    
                    if ( ( ! keep[i]) && (d_isolation[i]>0) )
                    {
                        ni = n_isolation[i];
                        // If criteria undef, then compute.
                        if (ni==-1)
                        {
                            ni = 0;
                            for (j=0 ; j<pp ; j++)
                            {
                                if ( DSS[i][j]<=d_isolation[i] ) ni++;
                            }
                            n_isolation[i] = ni;
                        }
                        // Keep biggest
                        if (ni>nmax)
                        {
                            nmax = ni;
                            iselect = i;
                        }
                    }
                    
                }// End for
            }
                break;
                //----------------------------------------
            case 5:
            {
                // Select point with highest density number
                
                nmax = 0;
                for (i=0 ; i<pp ; i++)
                {
                    if ( ( ! keep[i]) && (DTX[i]>0) )
                    {
                        ni = n_density[i];
                        // If criteria undef, then compute.
                        if (ni==-1)
                        {
                            ni = 0;
                            for (j=0 ; j<pp ; j++)
                            {
                                if ( DSS[i][j]<=DTX[i] ) ni++;
                            }
                            n_density[i] = ni;
                        }
                        // Keep biggest
                        if (ni>nmax)
                        {
                            nmax = ni;
                            iselect = i;
                        }
                    }
                }// End for
            }
                break;
                //----------------------------------------
            default:
                throw NOMAD::Exception ( "Sgtelib_Model_Search.cpp" ,
                                        __LINE__ ,"method index non valid" );
                //----------------------------------------
        }// End switch
        
        // If a point was selected,
        if ( (iselect>-1) & (!keep[iselect]) )
        {
            
            if ( bool_display ){
                out << "--> Selection of search point "<< iselect << std::endl;
            }
            // Note as selected
            keep[iselect] = true;
            nkeep++;
            failure = 0;
            
            // Update DT and d_isolation and, if needed, reset n_isolation.
            for (i=0 ; i<pp ; i++)
            {
                if (DT[i]>DSS[i][iselect])
                {
                    // Update DT
                    DT[i] = DSS[i][iselect];
                    DTX[i] = std::min(DTX[i],DT[i]);
                    n_density[i] = -1;
                    // Update delta
                    if (d_isolation[i]>DT[i])
                    {
                        // If d_isolation is updated, then n_isolation is reset
                        d_isolation[i] = DT[i];
                        n_isolation[i] = -1;
                    }
                }
            }
        }
        else
        {
            if ( bool_display ){
                out << "!! Method " << method << " did not return a point." << std::endl;
            }
            failure++;
        }
        
    }// End while
    
    
    // Free space
    delete [] d_isolation;
    delete [] n_isolation;
    delete [] n_density;
    delete [] DX;
    delete [] DT;
    delete [] DTX;
    for (i=0 ; i<pp ; i++) delete [] DSS[i];
    delete [] DSS;
    
    // Delete f and h
    delete [] f;
    delete [] h;
    
    //=============================================
    // Add the selected points
    //=============================================
    i = 0;
    NOMAD::Eval_Point * x;
    const NOMAD::Eval_Point * cur;
    cur = cache_surrogate.begin();
    while ( cur )
    {
        if ( keep[i] )
        {
            //out << "Selected point : " << *cur << std::endl;
            x = new NOMAD::Eval_Point (*cur);
            oracle_pts.push_back(x);
        }
        if (static_cast<int>(oracle_pts.size())>=nb_candidates)
            break;
        i++;
        cur = cache_surrogate.next();
    }
    delete [] keep;
    
    
    return true;
    
}//





/*------------------------------------------------------*/
/*  project and accept or reject an oracle trial point  */
/*  (private)                                           */
/*------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Search::check_oracle_point  ( const NOMAD::Cache   & cache          ,
                                                       const NOMAD::Point   & incumbent      ,
                                                       const NOMAD::Point   & delta_m        ,
                                                       const NOMAD::Display & out            ,
                                                       NOMAD::dd_type         display_degree ,
                                                       NOMAD::Point         & x                )
{
    bool proj_to_mesh = _p.get_model_search_proj_to_mesh();
    
    if ( display_degree == NOMAD::FULL_DISPLAY )
    {
        out << std::endl << "oracle candidate";
        if ( proj_to_mesh )
            out << " (before projection)";
        out << ": ( " << x << " )" << std::endl;
    }
    
    // projection to mesh:
    if ( proj_to_mesh )
    {
        x.project_to_mesh ( incumbent , delta_m , _p.get_lb() , _p.get_ub() );
        if ( display_degree == NOMAD::FULL_DISPLAY )
            out << "oracle candidate (after projection): ( " << x << " )" << std::endl;
    }
    
    // compare x and incumbent coordinates:
    if ( x == incumbent )
    {
        if ( display_degree == NOMAD::FULL_DISPLAY )
            out << "oracle candidate rejected (candidate==incumbent)" << std::endl;
        return false;
    }
    
    // two evaluations points are created in order to:
    //   1. check if the candidate is in cache
    //   2. have a prediction at x and at the incumbent:
    int n = x.size() , m = _p.get_bb_nb_outputs();
    
    NOMAD::Eval_Point * tk = new NOMAD::Eval_Point ( n , m ); // trial point
    tk->Point::operator = ( x );
    
    // check if the point is in cache:
    if ( cache.find ( *tk ) )
    {
        if ( display_degree == NOMAD::FULL_DISPLAY )
            out << "oracle candidate rejected (found in cache)" << std::endl;
        delete tk;
        return false;
    }
    else
    {
        if ( display_degree == NOMAD::FULL_DISPLAY )
            out << "oracle candidate is not in cache" << std::endl;
        return true;
    }
    
} // end check_oracle_points







/*--------------------------------------------------------*/
/*  insert a trial point in the evaluator control object  */
/*  (private)                                             */
/*--------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Search::register_point
( NOMAD::Point               x              ,
 NOMAD::Signature         & signature      ,
 const NOMAD::Point       & incumbent      ,
 NOMAD::dd_type             display_degree ) const
{
    int n = x.size();
    
    NOMAD::Eval_Point * tk = new NOMAD::Eval_Point ( n , _p.get_bb_nb_outputs() );
    NOMAD::Evaluator_Control * ev_control = _sgtelib_model_manager->get_evaluator_control();
    
    // if the search is optimistic, a direction is computed (this
    // will be used in case of success in the speculative search):
    if ( _p.get_model_search_optimistic() )
    {
        NOMAD::Direction dir ( n , 0.0 , NOMAD::MODEL_SEARCH_DIR );
        dir.Point::operator = ( x - incumbent );
        tk->set_direction  ( &dir );
    }
    
    tk->set_signature  ( &signature  );
    tk->Point::operator = ( x );
    
    if ( tk->get_bb_outputs().is_defined() )
    {
        throw NOMAD::Exception ( "Sgtelib_Model_Search.cpp" , __LINE__ ,
                                "register_point: point should not have defined bbo" );
    }
    
    // add the new point to the list of search trial points:
    ev_control->add_eval_point ( tk                      ,
                                display_degree          ,
                                _p.get_snap_to_bounds() ,
                                NOMAD::Double()         ,
                                NOMAD::Double()         ,
                                NOMAD::Double()         ,
                                NOMAD::Double()           );
    
}


/*---------------------------------------------------------------*/
/*                    optimize a model (private)                 */
/*---------------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Search::optimize_model ( const NOMAD::Cache      & cache          ,
                                                  const NOMAD::Point      & incumbent      ,
                                                  const NOMAD::Point      & delta_m        ,
                                                  const NOMAD::Eval_Point * x0s[4]         ,
                                                  const NOMAD::Display    & out            ,
                                                  NOMAD::dd_type            display_degree ,
                                                  std::vector<NOMAD::Point *>  & oracle_pts,
                                                  bool                    & stop           ,
                                                  NOMAD::stop_type        & stop_reason    )
{
    
    const bool bool_display = ( (string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"O")) || (display_degree == NOMAD::FULL_DISPLAY) );
    
    std::string error_str;
    bool        error = false;
    int         i , n = _p.get_dimension();
    int clock_start = static_cast<int>(std::clock());
    if ( bool_display )
    {
        out.open_block("Sgtelib_Model_Search::optimize_model");
        // Model evaluation budget
        out << "MAX_BB_EVAL: " << _p.get_SGTELIB_MODEL_EVAL_NB() << std::endl;
        // blackbox outputs:
        out << "BBOT: ";
        for ( i = 0 ; i < _p.get_bb_nb_outputs() ; i++ )
            out << _p.get_bb_output_type()[i] << " ";
        out << std::endl;
        // Formulation
        out << "Formulation: "
        << sgtelib_model_formulation_type_to_string ( _sgtelib_model_manager->get_formulation() )
        << std::endl;
    }
    // Reset stats
    _sgtelib_model_manager->reset_search_stats();
    
    // parameters creation:
    NOMAD::Parameters model_param ( out );
    
    // random seed:
    model_param.set_SEED ( _p.get_seed() + 10 * _all_searches_stats.get_MS_nb_searches() );
    // number of variables:
    model_param.set_DIMENSION ( n );
    
    
    model_param.set_BB_OUTPUT_TYPE ( _p.get_bb_output_type() );
    // blackbox inputs:
    model_param.set_BB_INPUT_TYPE ( _p.get_bb_input_type() );
    
    // barrier parameters:
    model_param.set_H_MIN  ( _p.get_h_min () );
    model_param.set_H_NORM ( _p.get_h_norm() );
    
    model_param.set_DISPLAY_ALL_EVAL (false);
    
    // starting points:
    for ( i = 0 ; i < 4 ; ++i )
        if ( x0s[i] )
            model_param.set_X0 ( *x0s[i] );
    
    // fixed variables:
    for ( i = 0 ; i < n ; ++i )
        if ( _p.variable_is_fixed(i) )
            model_param.set_FIXED_VARIABLE(i);
    
    // no model
    model_param.set_DISABLE_MODELS();
    
    // display:
    if (string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"I"))
    {
        model_param.set_DISPLAY_DEGREE ( NOMAD::NORMAL_DISPLAY );
        model_param.set_DISPLAY_STATS ("model_opt: BBE OBJ");
    }
    else
    {
        model_param.set_DISPLAY_DEGREE ( NOMAD::NO_DISPLAY );
    }
    
    // mesh: use isotropic mesh
    model_param.set_ANISOTROPIC_MESH ( false );
    model_param.set_MESH_UPDATE_BASIS ( 4.0 );
    model_param.set_MESH_COARSENING_EXPONENT ( 1 );
    model_param.set_MESH_REFINING_EXPONENT ( -1 );
    
    NOMAD::Point init_mesh_indices = _p.get_signature()->get_mesh()->get_mesh_indices();
    NOMAD::Point init_Delta = _p.get_signature()->get_mesh()->get_Delta();
    NOMAD::Point init_delta = _p.get_signature()->get_mesh()->get_delta();
    
    model_param.set_INITIAL_POLL_SIZE ( init_Delta , false );
    model_param.set_INITIAL_MESH_SIZE ( init_delta , false );
    
    
    
    //  model_param.set_INITIAL_MESH_INDEX ( 0 );
    
    // searches:
    model_param.set_LH_SEARCH ( int(_p.get_SGTELIB_MODEL_EVAL_NB()*0.3) , 0 );
    model_param.set_OPPORTUNISTIC_LH ( false );
    model_param.set_VNS_SEARCH ( true );
    model_param.set_SNAP_TO_BOUNDS ( true );
    
    // disable user calls:
    model_param.set_USER_CALLS_ENABLED ( false );
    
    // set flags:
    bool flag_check_bimads , flag_reset_mesh , flag_reset_barriers , flag_p1_active;
    NOMAD::Mads::get_flags ( flag_check_bimads   ,
                            flag_reset_mesh     ,
                            flag_reset_barriers ,
                            flag_p1_active        );
    
    NOMAD::Mads::set_flag_check_bimads   ( true  );
    NOMAD::Mads::set_flag_reset_mesh     ( true  );
    NOMAD::Mads::set_flag_reset_barriers ( true  );
    NOMAD::Mads::set_flag_p1_active      ( false );
    
    // bounds:
    model_param.set_LOWER_BOUND ( _sgtelib_model_manager->get_extended_lb() );
    model_param.set_UPPER_BOUND ( _sgtelib_model_manager->get_extended_ub() );
    
    //out << "Bounds : " << std::endl;
    //out << _sgtelib_model_manager->get_extended_lb() << std::endl;
    //out << _sgtelib_model_manager->get_extended_ub() << std::endl;
    
    
    // Max eval
    model_param.set_MAX_BB_EVAL ( _p.get_SGTELIB_MODEL_EVAL_NB() );
    
    // EXTERN SGTE
    if ( _p.get_SGTELIB_MODEL_FORMULATION() == NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
        model_param.set_BB_EXE( _p.get_SGTELIB_MODEL_DEFINITION() );
    
    // Declaration of the evaluator
    NOMAD::Evaluator * ev = NULL;
    
    // Declaration of the inner instance of mads
    NOMAD::Mads * mads_surrogate;
    
    // Best feasible/infeasible
    const NOMAD::Eval_Point * best_feas;
    const NOMAD::Eval_Point * best_infeas;
    
    
    //=============================
    // OPTIMIZATION OF THE MODEL
    //=============================
    try
    {
        // parameters validation:
        model_param.check();
        // model evaluator creation:
        if ( _p.get_SGTELIB_MODEL_FORMULATION() == NOMAD::SGTELIB_MODEL_FORMULATION_EXTERN )
            ev = new NOMAD::Evaluator ( model_param );
        else
            ev = new NOMAD::Sgtelib_Model_Evaluator ( model_param , _sgtelib_model_manager );
        // algorithm creation and execution:
        mads_surrogate = new NOMAD::Mads( model_param , ev );
        NOMAD::stop_type st = mads_surrogate->run();
        // check the stopping criterion:
        if ( st == NOMAD::CTRL_C || st == NOMAD::MAX_CACHE_MEMORY_REACHED )
        {
            std::ostringstream oss;
            oss << "model optimization: " << st;
            error_str   = oss.str();
            error       = true;
            stop        = true;
            stop_reason = st;
        }
        else if ( st == NOMAD::MAX_BB_EVAL_REACHED )
            _one_search_stats.add_MS_max_bbe();
        
        
        // update the stats on the number of model evaluations:
        _one_search_stats.update_MS_model_opt ( mads_surrogate->get_stats().get_bb_eval() );
        
        
        // get the solution(s):
        best_feas = mads_surrogate->get_best_feasible  ();
        best_infeas = mads_surrogate->get_best_infeasible();
        if ( bool_display )
        {
            out << "End of Model Optimization" << std::endl;
            if ( best_feas )
            {
                out << "Best feasible:";
                out << std::endl;
                out << "("<< (NOMAD::Point) *best_feas << ")" << std::endl;
                out << "f=" << best_feas->get_f() << " ; h=" << best_feas->get_h() << std::endl;
            }
            else if ( best_infeas )
            {
                out << "Best infeasible: ";
                out << std::endl;
                out << "("<< (NOMAD::Point) *best_infeas << ")" << std::endl;
                out << "f=" << best_infeas->get_f() << " ; h=" << best_infeas->get_h() << std::endl;
            }
        }
        
        
        
    }
    catch ( std::exception & e )
    {
        error     = true;
        error_str = std::string ( "optimization error: " ) + e.what();
        // out << error_str << std::endl;
        return false;
    }
    
    // reset flags:
    NOMAD::Mads::set_flag_check_bimads   ( flag_check_bimads   );
    NOMAD::Mads::set_flag_reset_mesh     ( flag_reset_mesh     );
    NOMAD::Mads::set_flag_reset_barriers ( flag_reset_barriers );
    NOMAD::Mads::set_flag_p1_active      ( flag_p1_active      );
    
    
    
    // Check existence of best feasible and best infeasible
    NOMAD::Point * xf = NULL , * xi = NULL;
    if ( best_feas )
    {
        xf = new NOMAD::Point ( *best_feas );
        delete _start_point_1;
        _start_point_1 = new NOMAD::Point ( *best_feas );
    }
    else if ( best_infeas )
    {
        xi = new NOMAD::Point ( *best_infeas );
    }
    if ( best_infeas )
    {
        delete _start_point_2;
        _start_point_2 = new NOMAD::Point ( *best_infeas );
    }
    if ( !xf && !xi )
    {
        error     = true;
        error_str = "optimization error: no solution";
        // out << error_str << std::endl;
        return false;
    }
    
    //======================================
    // SELECT CANDIDATES OUT OF THE CACHE
    //======================================
    
    i = _p.get_SGTELIB_MODEL_CANDIDATES_NB();
    const int nb_candidates = (i<=0) ? _p.get_bb_max_block_size() : i;
    
    
    if ( nb_candidates==1 )
    {
        if ( xf )
            oracle_pts.push_back(xf);
        else if ( xi )
            oracle_pts.push_back(xi);
    }
    else
    {
        if ( bool_display )
            out.open_block("Search filter");
        
        // Filter Cache_surrogate
        const double delta_m_norm = delta_m.norm().value();
        bool filter_ok = filter_cache ( out,
                                       cache,
                                       mads_surrogate->get_cache(),
                                       nb_candidates,
                                       delta_m_norm,
                                       oracle_pts );
        
        if ( bool_display )
        {
            out << "End of filter" << std::endl;
            out.close_block();
        }
        
        if ( ! filter_ok )
            throw NOMAD::Exception ( "Sgtelib_Model_Search.cpp" , __LINE__ ,"filter_cache failed." );
    }
    
    //==============================
    // PROJECTION
    //==============================
    if ( _p.get_model_search_proj_to_mesh() )
    {
        if ( bool_display && !string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"P") )
            out << "Projection of search candidates" << std::endl;
        
        std::vector<NOMAD::Point *>::iterator it;
        try {
            for (it = oracle_pts.begin() ; it != oracle_pts.end(); ++it){
                get_best_projection( cache         ,
                                    incumbent      ,
                                    delta_m        ,
                                    out            ,
                                    display_degree ,
                                    *ev            ,
                                    *it            );
            }
            
        }
        catch ( std::exception & e )
        {
            error     = true;
            error_str = std::string ( "get_best_projection error: " ) + e.what();
            out << error_str << std::endl;
        }
        
        
    }
    if ( ev )
        delete ev;
    if ( mads_surrogate )
        delete mads_surrogate;
    
    if ( bool_display )
    {
        out << "Error: " << error << std::endl;
        out << "Clock: " << double(std::clock() - clock_start)/double(CLOCKS_PER_SEC) << "sec" << std::endl;
        out.close_block();
    }
    
    return !error;
}






/*------------------------------------------------------*/
/*  get best projection                                 */
/*------------------------------------------------------*/
void NOMAD::Sgtelib_Model_Search::get_best_projection ( const NOMAD::Cache   & cache          ,
                                                       const NOMAD::Point   & incumbent      ,
                                                       const NOMAD::Point   & delta_m        ,
                                                       const NOMAD::Display & out            ,
                                                       NOMAD::dd_type         display_degree ,
                                                       NOMAD::Evaluator     & ev             ,
                                                       NOMAD::Point         * x             )
{
    if ( !x ) return;
    
    const bool bool_display = (string_find(_p.get_SGTELIB_MODEL_DISPLAY(),"P"));
    
    if (bool_display)
        out.open_block("Projection");
    
    const int n = x->size();
    const int m = _p.get_bb_nb_outputs();
    int i,j;
    
    
    
    // Copy of the mesh size
    double * dm = new double[n];
    for (j=0 ; j<n ; j++)
        dm[j] = delta_m.get_coord(j).value();
    
    
    // Build the set of trying points
    NOMAD::Eval_Point * x_try; // Projection Trial point
    NOMAD::Eval_Point * x_best; // Best Trial poin
    NOMAD::Double h, f, h_best, f_best;
    double dmj; // component i of delta_m;
    bool ce;
    
    
    // Non projected point
    //--------------------
    x_try = new NOMAD::Eval_Point ( n , m ); // trial point
    x_try->Point::operator = ( *x );
    ev.eval_x( *x_try , 0.0 , ce);
    _sgtelib_model_manager->check_hf(x_try);
    f = x_try->get_f();
    h = x_try->get_h();
    
    if ( bool_display )
    {
        out << "Non projected point:" << std::endl;
        out << "(" << (NOMAD::Point) *x_try << ")" << std::endl;
        out << "f=" << f << " ; h=" << h << std::endl;
    }
    
    delete x_try;
    x_try = NULL;
    
    
    // STD projected point
    //----------------
    x_try = new NOMAD::Eval_Point ( n , m );
    x_try->Point::operator = ( *x );
    // Project the point on the mesh
    x_try->project_to_mesh ( incumbent , delta_m , _p.get_lb() , _p.get_ub() );
    ev.eval_x( *x_try , 0.0 , ce);
    _sgtelib_model_manager->check_hf(x_try);
    f = x_try->get_f();
    h = x_try->get_h();
    
    if ( bool_display )
    {
        out << "Standard projected point:" << std::endl;
        out << "(" << (NOMAD::Point) *x_try << ")" << std::endl;
        out << "f=" << f << " ; h=" << h << std::endl;
    }
    
    x_best = x_try;
    f_best = f;
    h_best = h;
    x_try = NULL;
    
    
    // set of indexes for neighboors creation
    //----------------------------------------
    std::set<unsigned int> set_index;
    const int nb_voisins = static_cast<int>(pow(2.0,n));
    const int nb_proj_trial = 100*n;
    // Build the set of indexes
    if ( nb_proj_trial < nb_voisins*1.3 )
    {
        // Select randomly nb_proj_trial indexes
        for (i=0 ; i<nb_proj_trial ; i++)
            set_index.insert(NOMAD::RNG::rand()%nb_voisins);
    }
    else
    {
        // Use all indexes
        for (i=0 ; i<nb_voisins ; i++)
            set_index.insert(i);
    }
    
    
    
    // Build the set of neighboors
    //----------------------------
    
    std::set<NOMAD::Point> set_try;
    set_try.clear();
    
    if ( bool_display )
    {
        out.open_block("Projection candidates");
    }
    
    std::set<unsigned int>::iterator it_index; // Iterator in the set of indexes
    unsigned int index; // Index of the neighboor
    NOMAD::Point perturbation(x->size()); // Perturbation point
    
    
    // Try pertubation
    // ----------------------------
    for ( it_index=set_index.begin() ; it_index!=set_index.end() ; it_index++ )
    {
        
        // Compute perturbation
        index = *it_index;
        for ( j=0 ; j<x->size() ; j++ )
        {
            
            dmj = dm[j]; // Get the value of delta_m
            // Added 22 octobre 2015 to handle integer variables
            if ( (_p.get_bb_input_type())[j] == NOMAD::INTEGER  || (_p.get_bb_input_type())[j] == NOMAD::BINARY )
                dmj = 1.0;
            
            if ( index & 1 )
                dmj *= -1; // Inverse dmi depending on parity of index
            
            perturbation.set_coord(j,dmj); // Set perturbation
            index = (index >> 1); // Right shift (ie: divide by 2);
            
        }// End of the construction of the perturbation
        
        
        
        
        // Loop on points of the cache
        // ----------------------------
        const NOMAD::Eval_Point * xref = cache.begin();
        while ( xref )
        {
            
            // Build projection trial point
            // trial point
            
            NOMAD::Point * y = new NOMAD::Point ( n );
            y->Point::operator = ( *x+perturbation );
            y->project_to_mesh ( *xref , delta_m , _p.get_lb() , _p.get_ub() );
            
            
            
            // Rounding for integer and binary...
            for ( i=0 ; i<y->size() ; i++ )
            {
                if (  (_p.get_bb_input_type())[i] == NOMAD::INTEGER  )
                    (*y)[i] = (*y)[i].roundd();
                
                if (  (_p.get_bb_input_type())[i] == NOMAD::BINARY  )
                    (*y)[i] = ((*y)[i]>0.5)?1:0;
                
            }
            
            
            set_try.insert(*y);
            delete y;
            
            xref = cache.next();
            
        }// End of loop on the cache
        
    }// End of the construction of set_try
    
    
    
    
    if ( bool_display )
    {
        out << set_index.size() << " perturbation vectors" << std::endl;
        out << cache.size() << " cache pts" << std::endl;
        out << set_try.size() << " projection candidates!" << std::endl;
    }
    
    
    
    
    // Greedy selection
    //--------------------------------
    const int p = static_cast<int>( set_try.size() );
    bool * keep = new bool [p];
    
    if ( p<=nb_proj_trial )
    {
        // Not needed
        for ( i=0 ; i < p ; i++ )
            keep[i] = true;
    }
    else
    {
        // DO the greedy selection
        int inew;
        double * xnew_d;
        double * xref_d = new double [n];
        for ( j = 0 ; j < n ; j++ )
            xref_d[j] = x->get_coord(j).value();
        
        // Convert set of trial candidates into an array
        //--------------------------------
        double ** X = new double * [p];
        
        std::set<NOMAD::Point>::const_iterator it; // Iterator in the set of indexes
        i = 0;
        for ( it = set_try.begin() ; it != set_try.end() ; it++ )
        {
            X[i] = new double [n];
            for (j=0 ; j<n ; j++)
                X[i][j] = it->get_coord(j).value();
            
            i++;
        }
        
        // Distance to xref
        double * Dr  = new double [p];
        // Distance to the set of selected (or kept) points
        double * Ds  = new double [p];
        double * Ds2 = new double [p]; // Square of Ds
        
        // First selected point
        inew = 0;
        xnew_d = X[inew];
        double lambda = 3;
        
        // Buffer for dxj
        double dxj;
        
        
        // Initialise the distances
        //--------------------------------------
        for ( i = 0 ; i < p ; i++ )
        {
            // Compute distance between each point of the set and the point xref
            // and also between each point of the set and the selected point xnew
            Dr[i] = 0;
            Ds[i] = 0;
            for ( j = 0 ; j < n ; j++ )
            {
                dxj = (xref_d[j]-X[i][j])/dm[j];
                Dr[i] += dxj*dxj;
                dxj = (xnew_d[j]-X[i][j])/dm[j];
                Ds[i] += dxj*dxj;
            }
            Dr[i]  = sqrt(Dr[i]);
            Ds2[i] = Ds[i];
            Ds[i]  = sqrt(Ds[i]);
            // Init keep
            keep[i] = false;
        }
        
        // Note that we selected the first point
        int nb_keep = 1;
        keep[inew] = true;
        
        // Greedy selection
        //---------------------------
        
        // Q is the selection criteria: Q = Ds-lambda*Dr
        double Q_max, Q;
        // d is a buffer
        double d;
        
        while ( nb_keep < nb_proj_trial )
        {
            // Find the index that maximizes Ds-lambda*Dr
            Q_max = -INF;
            inew = 1;
            for ( i = 0 ; i < p ; i++ )
            {
                Q = Ds[i]-lambda*Dr[i];
                if ( Q > Q_max )
                {
                    inew = i;
                    Q_max = Q;
                }
            }
            
            if ( Ds[inew] == 0 )
            {
                // If the point is already in the set, then reduce lambda
                lambda*=0.9;
                if (lambda<1e-6)
                    break;
                
            }
            else
            {
                //Otherwise, add the point to the set
                keep[inew] = true;
                nb_keep++;
                xnew_d = X[inew];
                // Update its distance to the set
                Ds[inew] = 0;
                // Update the other distances to the set
                for ( i = 0 ; i < p ; i++ )
                {
                    if ( Ds[i] > 0 )
                    {
                        // Compute distance between each point of the set and the point xref
                        // and also between each point of the set and the selected point xnew
                        d = 0;
                        for ( j = 0 ; j < n ; j++ )
                        {
                            dxj = (xnew_d[j]-X[i][j])/dm[j];
                            d += dxj*dxj;
                        }
                        if ( Ds2[i] > d )
                        {
                            Ds2[i]=d;
                            Ds[i] = sqrt(d);
                        }
                    }
                }
            }
        }// End while
        
        
        if ( bool_display )
        {
            out << nb_keep << " projection candidates (after greedy selection)" << std::endl;
        }
        
        delete [] xref_d;
        delete [] Ds;
        delete [] Ds2;
        delete [] Dr;
        for ( i = 0 ; i < p ; i++ )
            delete [] X[i];
        delete [] X;
    }// End selection
    
    delete [] dm;
    
    
    // Evaluate projection trial points
    // in the surrogate model
    //-----------------------------------------
    std::set<NOMAD::Point>::const_iterator it_try; // Iterator in the set of indexes
    i = 0;
    for ( it_try = set_try.begin() ; it_try != set_try.end() ; it_try++ )
    {
        if ( keep[i] )
        {
            x_try = new NOMAD::Eval_Point ( n , m ); 
            x_try->Point::operator = ( *it_try );
            
            // Eval (with the same evaluator as for the model optimization)
            ev.eval_x( *x_try , 0.0 , ce);
            _sgtelib_model_manager->check_hf(x_try);
            f = x_try->get_f();
            h = x_try->get_h();
            
            if ( ( (h>0) && (h<h_best) ) || ( (h==0) && (f<f_best) ) )
            {
                
                if ( bool_display )
                {
                    out << "( " << x_try->get_bb_outputs() << " )" << std::endl;
                    out << "f =" << f << " ; h =" << h << " (new best projection)" << std::endl;
                }
                
                delete x_best;
                x_best = x_try;
                f_best = f;
                h_best = h;
                
            }
            else
                delete x_try;
            x_try = NULL;
        }
        i++;
    }
    
    
    
    x->Point::operator = ( *x_best );
    if ( bool_display )
    {
        out.close_block();
        out << "Selected candidate:  " << std::endl;
        out << "(" << (NOMAD::Point) *x << ")" << std::endl;
        out << "f=" << f_best << " ; h=" << h_best << std::endl;
        out << "( " << x_best->get_bb_outputs() << " )" << std::endl;
        out.close_block();
        
    }
    
    set_index.clear();
    set_try.clear();
    if ( x_best )
        delete x_best;
    if ( x_try )
        delete x_try;
    delete [] keep;
    
}//





// #endif
