/*------------------------------------------------------------------------------*/
/*  NOMAD - Nonlinear Optimization by Mesh Adaptive Direct search -             */
/*          version 3.8.1                                                       */
/*                                                                              */
/*  NOMAD - version 3.8.1 has been created by                                   */
/*                 Charles Audet        - Ecole Polytechnique de Montreal       */
/*                 Sebastien Le Digabel - Ecole Polytechnique de Montreal       */
/*                 Christophe Tribes    - Ecole Polytechnique de Montreal       */
/*                                                                              */
/*  The copyright of NOMAD - version 3.8.1 is owned by                          */
/*                 Sebastien Le Digabel - Ecole Polytechnique de Montreal       */
/*                 Christophe Tribes    - Ecole Polytechnique de Montreal       */
/*                                                                              */
/*  NOMAD v3 has been funded by AFOSR, Exxon Mobil, Hydro Québec, Rio Tinto     */
/*  and IVADO.                                                                  */
/*                                                                              */
/*  NOMAD v3 is a new version of NOMAD v1 and v2. NOMAD v1 and v2 were created  */
/*  and developed by Mark Abramson, Charles Audet, Gilles Couture, and John E.  */
/*  Dennis Jr., and were funded by AFOSR and Exxon Mobil.                       */
/*                                                                              */
/*  Contact information:                                                        */
/*    Ecole Polytechnique de Montreal - GERAD                                   */
/*    C.P. 6079, Succ. Centre-ville, Montreal (Quebec) H3C 3A7 Canada           */
/*    e-mail: nomad@gerad.ca                                                    */
/*    phone : 1-514-340-6053 #6928                                              */
/*    fax   : 1-514-340-5665                                                    */
/*                                                                              */
/*  This program is free software: you can redistribute it and/or modify it     */
/*  under the terms of the GNU Lesser General Public License as published by    */
/*  the Free Software Foundation, either version 3 of the License, or (at your  */
/*  option) any later version.                                                  */
/*                                                                              */
/*  This program is distributed in the hope that it will be useful, but WITHOUT */
/*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       */
/*  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License */
/*  for more details.                                                           */
/*                                                                              */
/*  You should have received a copy of the GNU Lesser General Public License    */
/*  along with this program. If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/*  You can find information on the NOMAD software at www.gerad.ca/nomad        */
/*------------------------------------------------------------------------------*/


/**
 \file   Multi_Obj_Evaluator.hpp
 \brief  NOMAD::Evaluator subclass for multiobjective optimization (headers)
 \author Sebastien Le Digabel
 \date   2010-04-20
 \see    Multi_Obj_Evaluator.cpp
 */
#ifndef __MULTI_OBJ_EVALUATOR__
#define __MULTI_OBJ_EVALUATOR__

#include "Phase_One_Evaluator.hpp"

namespace NOMAD {
    
    /// NOMAD::Evaluator subclass for multiobjective optimization.
    /**
     Version for two objective functions.
     */
    class Multi_Obj_Evaluator : public NOMAD::Evaluator {
        
    private:
        
        static int           _i1;   ///< Index of the first objective.
        static int           _i2;   ///< Index of the second objective.
        
        NOMAD::Double        _w1;   ///< Weight on the first objective function.
        NOMAD::Double        _w2;   ///< Weight on the second objective function.
        
        const NOMAD::Point * _ref;  ///< Reference point.
        
    public:
        
        /// Constructor.
        /**
         \param p Parameters -- \b IN.
         */
        Multi_Obj_Evaluator ( const NOMAD::Parameters & p )
        : NOMAD::Evaluator ( p    ) ,
        _w1              ( 1.0  ) ,
        _w2              ( 0.0  ) ,
        _ref             ( NULL )   { _is_multi_obj = true; }
        
        /// Destructor.
        virtual ~Multi_Obj_Evaluator ( void ) {}
        
        /// Initialization of objective indexes.
        /**
         \param index_obj List of objective indexes -- \b IN.
         */
        static void set_obj_indexes ( const std::list<int> & index_obj );
        
        /// Updates after a MADS run.
        /**
         This virtual method is called every time a MADS run is terminated.
         \param stats        Stats                           -- \b IN.
         \param ev_control   Evaluator control               -- \b IN.
         \param true_barrier Barrier for true functions      -- \b IN.
         \param sgte_barrier Barrier for surrogate functions -- \b IN.
         \param pareto_front Pareto front                    -- \b IN.
         */
        virtual void update_mads_run ( const NOMAD::Stats             & stats        ,
                                      const NOMAD::Evaluator_Control & ev_control   ,
                                      const NOMAD::Barrier           & true_barrier ,
                                      const NOMAD::Barrier           & sgte_barrier ,
                                      const NOMAD::Pareto_Front      & pareto_front   ) {}
        
        /// Compute \c f(x) from the blackbox outputs of a point.
        /**
         - Bi-objective version.
         - Computation of \c f taking into account the two objectives
         with a reformulation based on a reference point, or
         with weights when no reference is available.
         \param x The evaluation point -- \b IN/OUT.
         */
        virtual void compute_f ( NOMAD::Eval_Point & x ) const;
        
        /// Get the index of the first objective function.
        /**
         \return The index of the first objective function.
         */
        static int get_i1 ( void ) { return _i1; }
        
        /// Get the index of the second objective function.
        /**
         \return The index of the second objective function.
         */
        static int get_i2 ( void ) { return _i2; }
        
        /// Set the weights.
        /**
         \param w1 Weight on the first objective function  -- \b IN.
         \param w2 Weight on the second objective function -- \b IN.
         */
        void set_weights ( const NOMAD::Double & w1 ,
                          const NOMAD::Double & w2   )
        {
            _w1 = w1;
            _w2 = w2;
        }
        
        /// Set the reference point.
        /**
         \param ref A pointer to the reference point -- \b IN.
         */
        void set_ref ( const NOMAD::Point * ref ) { _ref = ref; }
        
        
    };
}

#endif
