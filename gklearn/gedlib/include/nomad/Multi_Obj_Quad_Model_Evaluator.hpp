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
 \file   Multi_Obj_Quad_Model_Evaluator.hpp
 \brief  NOMAD::Multi_Obj_Evaluator subclass for quadratic model optimization (headers)
 \author Christophe Tribes 
 \date   2014-06-19
 \see    Multi_Obj_Quad_Model_Evaluator.cpp
 */
#ifndef __MULTI_OBJ_QUAD_MODEL_EVALUATOR__
#define __MULTI_OBJ_QUAD_MODEL_EVALUATOR__

#include "Quad_Model_Evaluator.hpp"
#include "Multi_Obj_Evaluator.hpp"

namespace NOMAD {
	
	/// Bi-objective NOMAD::Evaluator subclass for quadratic model.
	class Multi_Obj_Quad_Model_Evaluator : public NOMAD::Quad_Model_Evaluator, public NOMAD::Multi_Obj_Evaluator {     		
	public:
		
		/// Constructor.
		/**
		 \param p     Parameters -- \b IN.
		 \param model Model      -- \b IN.
		 */
		Multi_Obj_Quad_Model_Evaluator ( const NOMAD::Parameters & p     ,
										const NOMAD::Quad_Model & model   ) : NOMAD::Quad_Model_Evaluator(p,model),NOMAD::Multi_Obj_Evaluator(p) {_is_model_evaluator=true;}
		
		/// Destructor.
		virtual ~Multi_Obj_Quad_Model_Evaluator ( void ){;}
		
		
		///  Evaluate the blackboxes quad model at a given trial point
		/**
		 \param x          point to evaluate     -- \b IN/OUT.
		 \param h_max      h_max for barrier     -- \b IN.
		 \param count_eval Count eval if true    -- \b IN.
		 \return A boolean equal to \c false if the evaluation failed.
		 */	
		virtual bool eval_x ( NOMAD::Eval_Point   & x          ,
							 const NOMAD::Double & h_max      ,
							 bool                & count_eval   ) const {return Quad_Model_Evaluator::eval_x(x,h_max,count_eval);}
		
		
	};
}

#endif

