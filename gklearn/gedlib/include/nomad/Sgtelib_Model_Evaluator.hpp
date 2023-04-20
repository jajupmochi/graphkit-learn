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
 \file   Sgtelib_Model_Evaluator.hpp
 \brief  Interface between nomad evaluator and Sgtelib_Model_Manager.
 \author Bastien Talgorn
 \date   2013-04-25
 \see    Sgtelib_Model_Manager.cpp
 */

#ifndef __SGTELIB_MODEL_EVALUATOR__
#define __SGTELIB_MODEL_EVALUATOR__


#include "Sgtelib_Model_Manager.hpp"
#include "Search.hpp"
#include "Evaluator.hpp"


namespace NOMAD {
    
    /// NOMAD::Evaluator subclass for quadratic model optimization.
    class Sgtelib_Model_Evaluator : public NOMAD::Evaluator {
        
    private:
        
        NOMAD::Sgtelib_Model_Manager * _sgtelib_model_manager; ///< The sgtelib_model model.
        
    public:
        
        /// Constructor.
        Sgtelib_Model_Evaluator (  const NOMAD::Parameters &      p ,
                                 NOMAD::Sgtelib_Model_Manager * sgtelib_model_manager )
        : NOMAD::Evaluator     ( p                     ) ,
        _sgtelib_model_manager ( sgtelib_model_manager ) {}
        
        
        /// Destructor.
        virtual ~Sgtelib_Model_Evaluator ( void ) {}
        
        /// Evaluate the blackboxes at a given trial point.
        virtual bool eval_x ( NOMAD::Eval_Point   & x          ,
                             const NOMAD::Double & h_max      ,
                             bool                & count_eval   ) const;
        
    };
}

#endif

// #endif
