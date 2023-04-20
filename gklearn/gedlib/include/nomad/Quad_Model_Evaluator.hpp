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
 \file   Quad_Model_Evaluator.hpp
 \brief  NOMAD::Evaluator subclass for quadratic model optimization (headers)
 \author Sebastien Le Digabel
 \date   2010-08-31
 \see    Quad_Model_Evaluator.cpp
 */
#ifndef __QUAD_MODEL_EVALUATOR__
#define __QUAD_MODEL_EVALUATOR__

#include "Search.hpp"


namespace NOMAD {
    
    /// NOMAD::Evaluator subclass for quadratic model optimization.
    class Quad_Model_Evaluator  {
        
    private:
        
        int       _n;           ///< Number of variables.
        int       _nm1;         ///< Number of variables minus one.
        int       _m;           ///< Number of blackbox outputs.
        double  * _x;           ///< An evaluation point.
        double ** _alpha;       ///< Model parameters.
        bool      _model_ready; ///< \c true if model ready to evaluate.
        
    public:
        
        /// Constructor.
        /**
         \param p     Parameters -- \b IN.
         \param model Model      -- \b IN.
         */
        Quad_Model_Evaluator ( const NOMAD::Parameters & p     ,
                              const NOMAD::Quad_Model & model   );
        
        /// Destructor.
        virtual ~Quad_Model_Evaluator ( void );
        
        /// Evaluate the blackboxes at a given trial point.
        /**
         \param x The trial point -- \b IN/OUT.
         \param h_max      Maximal feasibility value \c h_max -- \b IN.
         \param count_eval Flag indicating if the evaluation has to be counted
         or not -- \b OUT.
         \return A boolean equal to \c false if the evaluation failed.
         */
        virtual bool eval_x ( NOMAD::Eval_Point   & x          ,
                             const NOMAD::Double & h_max      ,
                             bool                & count_eval   ) const;
        
        /// Evaluate the blackbox functions at a given trial point (#2).
        /**
         - Non-const version.
         - Calls the const version by default.
         - May be user-defined.
         - Surrogate or true evaluation depending on the value of \c x.is_surrogate().
         \param x          The trial point                    -- \b IN/OUT.
         \param h_max      Maximal feasibility value \c h_max -- \b IN.
         \param count_eval Flag indicating if the evaluation has to be counted
         or not -- \b OUT.
         \return A boolean equal to \c false if the evaluation failed.
         */
        virtual bool eval_x ( NOMAD::Eval_Point   & x          ,
                             const NOMAD::Double & h_max      ,
                             bool                & count_eval   )
        {
            return static_cast<const NOMAD::Quad_Model_Evaluator *>(this)->eval_x ( x , h_max, count_eval );
        }
        
        
        
        
        /// Evaluate the gradient of a blackboxe at a given trial point.
        /**
         \param x The trial point -- \b IN/OUT.
         \param g The gradient of a bb model at the trial point \c x -- \b OUT.
         \param output_index The index of the black box.           -- \b IN.
         \param count_eval Flag indicating if the evaluation has to be counted
         or not -- \b OUT.
         \return A boolean equal to \c false if the evaluation failed.
         */
        
        virtual bool evalGrad_x (const NOMAD::Point   & x   ,
                                 NOMAD::Point   & g         ,
                                 const int & output_index   ,
                                 bool                & count_eval   ) const;
        
    };
}

#endif
