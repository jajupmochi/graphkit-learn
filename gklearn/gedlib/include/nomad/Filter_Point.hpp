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
  \file   Filter_Point.hpp
  \brief  Point stored in the filter (headers)
  \author Sebastien Le Digabel
  \date   2010-04-09
*/
#ifndef __FILTER_POINT__
#define __FILTER_POINT__

#include "Eval_Point.hpp"
#include "Set_Element.hpp"

namespace NOMAD {

  /// Class for the representation of NOMAD::Eval_Point objects stored in the filter.
  class Filter_Point : public NOMAD::Set_Element<NOMAD::Eval_Point> {

  private:

    /// Affectation operator.
    /**
       \param fp The right-hand side object -- \b IN.
       \return \c *this as the result of the affectation.
    */
    Filter_Point & operator = ( const Filter_Point & fp );

  public:
  
    /// Constructor.
    /**
       \param ep A pointer to the NOMAD::Eval_Point object that
                 is stored in the filter -- \b IN.
    */
    Filter_Point ( const NOMAD::Eval_Point * ep )
      : NOMAD::Set_Element<NOMAD::Eval_Point> ( ep ) {}

    /// Copy constructor.
    /**
       \param fp The copied object -- \b IN.
    */
    explicit Filter_Point ( const Filter_Point & fp )
      : NOMAD::Set_Element<NOMAD::Eval_Point> ( fp.get_element() ) {}

    /// Destructor.
    virtual ~Filter_Point ( void ) {}

    /// Comparison operator.
    /**
       \param  fp The right-hand side object.
       \return A boolean equal to \c true if \c *this \c < \c fp.
    */
    virtual bool operator < ( const NOMAD::Set_Element<NOMAD::Eval_Point> & fp ) const
    {
      return get_element()->get_h().value() < fp.get_element()->get_h().value();
    }

    /// Access to the point.
    /**
       \return A pointer to the NOMAD::Eval_Point stored in the cache.
    */
    const NOMAD::Eval_Point * get_point ( void ) const { return get_element(); }
  };
}
#endif
