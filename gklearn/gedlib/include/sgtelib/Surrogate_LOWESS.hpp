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

#ifndef __SGTELIB_SURROGATE_LOWESS__
#define __SGTELIB_SURROGATE_LOWESS__

#include "Surrogate.hpp"

#include <iostream>
#include <fstream>

namespace SGTELIB {

  /*--------------------------------------*/
  /*         Surrogate_LOWESS class        */
  /*--------------------------------------*/
  class Surrogate_LOWESS : public SGTELIB::Surrogate {

  protected:

    int _q; // Number of basis functions
    int _q_old; 
    int _degree; // Degree of local regression
    double ** _H; // Design matrix
    double *  _W; // Weights of each observation
    double ** _A; // Matrix of the linear system (and preconditionner)
    double ** _HWZ; // Second term
    double * _u; // First line of inverse of A
    double * _old_u; // Last value of gamma
    double * _old_x; // Last value of x

    SGTELIB::Matrix _ZZsi; // Outputs for one point (buffer)

    // build model (private):
    virtual bool build_private (void);

    void predict_private ( const SGTELIB::Matrix & XXs,
                                 SGTELIB::Matrix * ZZs); 

    void delete_matrices (void);

    void predict_private_single ( SGTELIB::Matrix XXs , int i_exclude = -1);

    // Compute metrics
    const SGTELIB::Matrix * get_matrix_Zvs (void);

  public:
    // Constructor
    Surrogate_LOWESS ( SGTELIB::TrainingSet & trainingset ,   
                    SGTELIB::Surrogate_Parameters param) ;

    // destructor:
    virtual ~Surrogate_LOWESS ( void );

    // Build the monome exponents
    virtual void display_private ( std::ostream & out ) const;

  };
}

#endif
