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
 \file   Sgtelib_Model_Evaluator.cpp
 \brief  Interface between nomad evaluator and Sgtelib_Model_Manager.
 \author Bastien Talgorn
 \date   2013-04-25
 \see    Sgtelib_Model_Manager.cpp
 */

#include "Sgtelib_Model_Evaluator.hpp"


/*------------------------------------------------------------------------*/
/*                evaluate the sgtelib_model model at a given trial point */
/*------------------------------------------------------------------------*/
bool NOMAD::Sgtelib_Model_Evaluator::eval_x ( NOMAD::Eval_Point   & x          ,
                                             const NOMAD::Double & h_max      ,
                                             bool                & count_eval   ) const
{
    return _sgtelib_model_manager->eval_x( &x         ,
                                          h_max      ,
                                          count_eval );
}


