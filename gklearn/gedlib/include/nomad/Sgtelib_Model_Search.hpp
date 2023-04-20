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

#ifndef __SGTELIB_MODEL_SEARCH__
#define __SGTELIB_MODEL_SEARCH__

#include "LH_Search.hpp"
#include "Sgtelib_Model_Evaluator.hpp"
#include "Sgtelib_Model_Manager.hpp"
#include <time.h>

namespace NOMAD {
    
    /// Model search.
    class Sgtelib_Model_Search : public NOMAD::Search , private NOMAD::Uncopyable {
        
    private:
        
        NOMAD::Model_Stats _one_search_stats;   ///< Stats for one search.
        NOMAD::Model_Stats _all_searches_stats; ///< Stats for all searches.
        
        NOMAD::Sgtelib_Model_Manager * _sgtelib_model_manager;
        
        std::vector<NOMAD::bb_output_type> _bbot; ///< Blackbox output types.
        
        NOMAD::Point * _start_point_1;
        NOMAD::Point * _start_point_2;
        /// The non projected result of a surrogate model optimization
        // is memorized and used as ONE of the starting points in the next surrogate model optim.
        
        /// Delete a list of points.
        /**
         \param pts The points -- \b IN/OUT.
         */
        static void clear_pts ( std::vector<NOMAD::Point *> & pts );
        
        /// Delete a list of evaluation points.
        /**
         \param pts The points -- \b IN/OUT.
         */
        static void clear_pts ( std::vector<NOMAD::Eval_Point *> & pts );
        
        /// Create oracle points by optimizing the model.
        /**
         \param  mads           The mads instance                         -- \b IN.
         \param  incumbent      The incumbent                             -- \b IN.
         \param  delta_m        Mesh size parameter                       -- \b IN.
         \param  out            The NOMAD::Display object                 -- \b IN.
         \param  display_degree Display degree                            -- \b IN.
         \param  display_lim    Max number of pts when sets are displayed -- \b IN.
         \param  oracle_pts     Oracle candidates points                  -- \b OUT.
         \param  stop           Stop flag                                 -- \b OUT.
         \param  stop_reason    Stop reason                               -- \b OUT.
         \return A boolean equal to \c true if oracle points are proposed.
         */
        bool create_oracle_pts
        (
         const NOMAD::Mads                      & mads           ,
         const NOMAD::Point                     & incumbent      ,
         const NOMAD::Point                     & delta_m        ,
         const NOMAD::Display                   & out            ,
         NOMAD::dd_type                           display_degree ,
         int                                      display_lim    ,
         std::vector<NOMAD::Point *>            & oracle_pts     ,
         bool                                   & stop           ,
         NOMAD::stop_type                       & stop_reason      );
        
        /// Model optimization.
        /**
         \param cache          The cache of points       -- \b IN.
         \param incumbent      The current incumbent     -- \b IN.
         \param delta_m        Mesh size parameter       -- \b IN.
         \param x0s            The three starting points -- \b IN.
         \param out            The NOMAD::Display object -- \b IN.
         \param display_degree Display degree            -- \b IN.
         \param oracle_pts     Oracle candidates points  -- \b OUT.
         \param stop           Stop flag                 -- \b OUT.
         \param stop_reason    Stop reason               -- \b OUT.
         \return A boolean equal to \c true if oracle points are proposed.
         */
        bool optimize_model
        (
         const NOMAD::Cache            & cache          ,
         const NOMAD::Point            & incumbent      ,
         const NOMAD::Point            & delta_m        ,
         const NOMAD::Eval_Point       * x0s[3]         ,
         const NOMAD::Display          & out            ,
         NOMAD::dd_type                 display_degree  ,
         std::vector<NOMAD::Point *>   & oracle_pts     ,
         bool                          & stop           ,
         NOMAD::stop_type              & stop_reason      );
        
        /// Cache filtering.
        /**
         \param cache           The cache of points                 -- \b IN.
         \param cache_surrogate The cache of surrogate points       -- \b IN.
         \param out             The NOMAD::Display object           -- \b IN.
         \param nb_candidates   The number of candidates            -- \b IN.
         \param delta_m_norm    The distance for filtering          -- \b IN.
         \param oracle_pts      Oracle candidates points            -- \b OUT.
         \return A boolean equal to \c true if oracle points are proposed.
         */
        bool filter_cache
        (
         const NOMAD::Display        & out             ,
         const NOMAD::Cache          & cache           ,
         const NOMAD::Cache          & cache_surrogate ,
         const int                   & nb_candidates   ,
         const double                & delta_m_norm    ,
         std::vector<NOMAD::Point *> & oracle_pts      );
        
        
        /// Get best projection
        /**
         \param  cache          Cache of true evaluations -- \b IN.
         \param  incumbent      The incumbent             -- \b IN.
         \param  delta_m        Mesh size parameter       -- \b IN.
         \param  out            The NOMAD::Display object -- \b IN.
         \param  display_degree Display degree            -- \b IN.
         \param  ev             The Evaluator             -- \b IN.
         \param  x              The oracle point          -- \b IN/OUT.
         */
        void get_best_projection
        (
         const NOMAD::Cache   & cache           ,
         const NOMAD::Point   & incumbent       ,
         const NOMAD::Point   & delta_m         ,
         const NOMAD::Display & out             ,
         NOMAD::dd_type         display_degree  ,
         NOMAD::Evaluator     & ev              ,
         NOMAD::Point         * x                );
        
        
        /// Project and accept or reject an oracle trial point.
        /**
         \param  cache          Cache of true evaluations -- \b IN.
         \param  incumbent      The incumbent             -- \b IN.
         \param  delta_m        Mesh size parameter       -- \b IN.
         \param  out            The NOMAD::Display object -- \b IN.
         \param  display_degree Display degree            -- \b IN.
         \param  x              The oracle point         -- \b IN/OUT.
         \return A boolean equal to \c true if the point is accepted.
         */
        bool check_oracle_point
        (
         const NOMAD::Cache   & cache          ,
         const NOMAD::Point   & incumbent      ,
         const NOMAD::Point   & delta_m        ,
         const NOMAD::Display & out            ,
         NOMAD::dd_type         display_degree ,
         NOMAD::Point         & x                );
        
        
        /// Insert a trial point in the evaluator control object.
        /**
         \param x              The point coordinates               -- \b IN.
         \param signature      Signature                           -- \b IN.
         \param incumbent      The incumbent                       -- \b IN.
         \param display_degree Display degree                      -- \b IN.
         */
        void register_point
        (
         NOMAD::Point               x              ,
         NOMAD::Signature         & signature      ,
         const NOMAD::Point       & incumbent      ,
         NOMAD::dd_type             display_degree ) const;
        
        /*----------------------------------------------------------------------*/
        
    public:
        
        /// Constructor.
        /**
         \param p Parameters -- \b IN.
         */
        Sgtelib_Model_Search ( NOMAD::Parameters & p )
        : NOMAD::Search ( p , NOMAD::MODEL_SEARCH ) , _sgtelib_model_manager (NULL)
        {
            _bbot = p.get_bb_output_type();
            
        }
        
        /// Destructor.
        virtual ~Sgtelib_Model_Search ( void ) { reset(); }
        
        /// Reset.
        virtual void reset ( void );
        
        /// Give a link to ev_control
        void set_sgtelib_model_manager ( NOMAD::Sgtelib_Model_Manager * sgtelib_model_manager )
        {
            _sgtelib_model_manager = sgtelib_model_manager;
            _start_point_1 = NULL;
            _start_point_2 = NULL;
        }
        
        
        /// The sgtelib_model model search.
        /**
         Based on quadratic regression/MFN interpolation models.
         \param mads           NOMAD::Mads object invoking this search -- \b IN/OUT.
         \param nb_search_pts  Number of generated search points       -- \b OUT.
         \param stop           Stop flag                               -- \b IN/OUT.
         \param stop_reason    Stop reason                             -- \b OUT.
         \param success        Type of success                         -- \b OUT.
         \param count_search   Count or not the search                 -- \b OUT.
         \param new_feas_inc   New feasible incumbent                  -- \b IN/OUT.
         \param new_infeas_inc New infeasible incumbent                -- \b IN/OUT.
         */
        virtual void search
        ( NOMAD::Mads              & mads           ,
         int                      & nb_search_pts  ,
         bool                     & stop           ,
         NOMAD::stop_type         & stop_reason    ,
         NOMAD::success_type      & success        ,
         bool                     & count_search   ,
         const NOMAD::Eval_Point *& new_feas_inc   ,
         const NOMAD::Eval_Point *& new_infeas_inc   );
        
        
        //// Display stats.
        /**
         \param out The NOMAD::Display object -- \b IN.
         */
        virtual void display ( const NOMAD::Display & out ) const
        {
            out << _all_searches_stats;
        }
    };
}

#endif
// #endif
