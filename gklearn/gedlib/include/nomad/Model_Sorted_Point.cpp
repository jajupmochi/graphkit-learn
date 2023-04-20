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
 \file   Model_Sorted_Point.cpp
 \brief  Interpolation point with distance to model center (implementation)
 \author Sebastien Le Digabel
 \date   2010-11-15
 \see    Model_Sorted_Point.hpp
 */
#include "Model_Sorted_Point.hpp"

/*---------------------------------------------------------*/
/*                        constructor                      */
/*---------------------------------------------------------*/
NOMAD::Model_Sorted_Point::Model_Sorted_Point ( NOMAD::Point * x ,
                                               const NOMAD::Point & center ) : _x(x)
{
    int i , n = center.size();
    if ( x && x->size() == n )
    {
        _dist = 0.0;
        for ( i = 0 ; i < n ; ++i )
            if ( (*x)[i].is_defined() && center[i].is_defined() )
            {
                _dist += ( (*x)[i] - center[i] ).pow2();
            }
            else
            {
                
                _dist.clear();
                break;
            }
    }
}

/*---------------------------------------------------------*/
/*                   affectation operator                  */
/*---------------------------------------------------------*/
NOMAD::Model_Sorted_Point & NOMAD::Model_Sorted_Point::operator = ( const NOMAD::Model_Sorted_Point & x )
{
    _x    = x._x;
    _dist = x._dist;
    return *this;
}

/*---------------------------------------------------------*/
/*                    comparison operator                  */
/*---------------------------------------------------------*/
bool NOMAD::Model_Sorted_Point::operator < ( const Model_Sorted_Point & x ) const
{
    if ( _dist.is_defined() )
    {
        if ( !x._dist.is_defined() )
            return true;
        return _dist < x._dist;
    }
    return false;
}
