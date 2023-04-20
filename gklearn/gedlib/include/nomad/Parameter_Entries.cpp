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
 \file   Parameter_Entries.cpp
 \brief  Parameter entries (implementation)
 \author Sebastien Le Digabel
 \date   2010-04-05
 \see    Parameter_Entries.hpp
 */
#include "Parameter_Entries.hpp"

/*--------------------------------------------*/
/*                 destructor                 */
/*--------------------------------------------*/
NOMAD::Parameter_Entries::~Parameter_Entries ( void )
{
    std::multiset<NOMAD::Parameter_Entry*, NOMAD::Parameter_Entry_Comp>::iterator
    end = _entries.end() , it;
    for ( it = _entries.begin() ; it != end ; ++it)
        delete *it;
}

/*--------------------------------------------*/
/*      finds a specific entry in the set     */
/*--------------------------------------------*/
NOMAD::Parameter_Entry * NOMAD::Parameter_Entries::find ( const std::string & name ) const
{
    NOMAD::Parameter_Entry p (name);
    std::multiset<NOMAD::Parameter_Entry*, NOMAD::Parameter_Entry_Comp>::const_iterator
    it = _entries.find ( &p );
    if ( it != _entries.end() )
        return (*it);
    return NULL;
}

/*----------------------------------------*/
/*      inserts an entry into the set     */
/*----------------------------------------*/
void NOMAD::Parameter_Entries::insert ( NOMAD::Parameter_Entry * entry )
{
    NOMAD::Parameter_Entry * cur = find ( entry->get_name() );
    if ( cur )
    {
        entry->set_unique ( false );
        cur->set_unique   ( false );
        while ( cur->get_next() )
            cur = cur->get_next();
        cur->set_next ( entry );
    }
    _entries.insert ( entry );
}

/*----------------------------------------*/
/*       find a non-interpreted entry     */
/*----------------------------------------*/
NOMAD::Parameter_Entry * NOMAD::Parameter_Entries::find_non_interpreted ( void ) const
{
    std::multiset<NOMAD::Parameter_Entry*, NOMAD::Parameter_Entry_Comp>::const_iterator
    end = _entries.end() , it;
    for ( it = _entries.begin() ; it != end ; ++it )
        if ( !(*it)->has_been_interpreted() )
            return *it;
    return NULL;
}

/*--------------------------------------------*/
/*                   display                  */
/*--------------------------------------------*/
void NOMAD::Parameter_Entries::display ( const NOMAD::Display & out ) const
{
    std::multiset<NOMAD::Parameter_Entry*,NOMAD::Parameter_Entry_Comp>::const_iterator
    end = _entries.end() , it;
    for ( it = _entries.begin() ; it != end ; ++it )
        out << **it << std::endl;
}
