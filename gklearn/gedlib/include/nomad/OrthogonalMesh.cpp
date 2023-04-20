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
 \file   OrthogonalMesh.hpp
 \brief  implementation
 \author Christophe Tribes
 \date   2014-06-19
 \see    SMesh.cpp XMesh.cpp
 */

#include "OrthogonalMesh.hpp"

/// Constructor (called only by derived objects).
NOMAD::OrthogonalMesh::OrthogonalMesh (bool                   anisotropic_mesh ,
                                       const NOMAD::Point    & Delta_0   ,
                                       const NOMAD::Point    & Delta_min ,
                                       const NOMAD::Point    & delta_min ,
                                       const NOMAD::Point   & fixed_variables ,
                                       const NOMAD::Point   & granularity ,
                                       NOMAD::Double            update_basis,
                                       int                     coarsening_step,
                                       int                     refining_step,
                                       int                   limit_mesh_index ) :
_anisotropic_mesh   ( anisotropic_mesh ),
_delta_0            ( Delta_0 ),
_Delta_0            ( Delta_0 ),
_Delta_min          ( Delta_min ),
_delta_min          ( delta_min ),
_fixed_variables    ( fixed_variables ),
_granularity        ( granularity ),
_update_basis       ( update_basis ),
_coarsening_step    ( coarsening_step ),
_refining_step      ( refining_step ),
_limit_mesh_index   ( limit_mesh_index )
{
    
    
    _Delta_min_is_defined = _Delta_min.is_defined();
    _Delta_min_is_complete = _Delta_min.is_complete();
    
    _delta_min_is_defined = _delta_min.is_defined();
    _delta_min_is_complete = _delta_min.is_complete();
    
    _n = Delta_0.size();
    
    _n_free_variables = _n - _fixed_variables.nb_defined();
    
    if ( _granularity.is_defined() && ( ! _granularity.is_complete() || _granularity.size() != _n ) )
        throw NOMAD::Exception ( "OrthogonalMesh.hpp" , __LINE__ ,
                                "NOMAD::OrthogonalMesh::OrthogonalMesh(): granularity has undefined values" );
    
    if ( !_Delta_0.is_complete() )
        throw NOMAD::Exception (  "OrthogonalMesh.hpp" , __LINE__ ,
                                "NOMAD::OrthogonalMesh::OrthogonalMesh(): delta_0 has undefined values" );
    
    if ( _delta_min_is_defined && delta_min.size() != _n )
        throw NOMAD::Exception ( "OrthogonalMesh.hpp" , __LINE__ ,
                                "NOMAD::OrthogonalMesh::OrthogonalMesh(): delta_0 and delta_min have different sizes" );
    
    if ( _Delta_min_is_defined && Delta_min.size() != _n )
        throw NOMAD::Exception ( "OrthogonalMesh.hpp" , __LINE__ ,
                                "NOMAD::OrthogonalMesh::OrthogonalMesh(): Delta_0 and Delta_min have different sizes" );
    
    
    std::string error;
    _all_granular = ( _granularity.is_defined() && _granularity.is_complete() ) ? true:false ;
    for ( int k = 0 ; k < _n ; ++k )
    {
        // we check that Delta_min <= Delta_0 and that delta_min <= delta_0:
        if ( _delta_min_is_defined &&
            _delta_min[k].is_defined()                        &&
            _delta_0[k] < _delta_min[k]        )
        {
            error = "NOMAD::OrthogonalMesh::OrthogonalMesh(): delta_0 < delta_min";
            break;
        }
        if ( _Delta_min_is_defined &&
            _Delta_min[k].is_defined()                        &&
            _Delta_0[k] < _Delta_min[k]     )
        {
            error = "NOMAD::OrthogonalMesh::OrthogonalMesh(): Delta_0 < Delta_min";
            break;
        }
        
        if ( _all_granular && _granularity[k] == 0 )
            _all_granular = false;
        
    }
    
    if ( !error.empty() )
        throw NOMAD::Exception ( "OrthogonalMesh.hpp" , __LINE__ , error );
}



bool NOMAD::OrthogonalMesh::is_finer_than_initial (void) const
{
    NOMAD::Point delta;
    get_delta(delta);
    
    for (int i =0; i < _n ; ++i )
        if ( !_fixed_variables[i].is_defined() && delta[i] >= _delta_0[i] )
            return false;
    
    return true;
}



/// Manually set the min mesh size per coordinate.
void NOMAD::OrthogonalMesh::set_min_mesh_sizes ( const NOMAD::Point & delta_min )
{
    
    // If delta_min undefined than _delta_min->undefined
    if ( ! delta_min.is_defined() )
    {
        _delta_min.clear();
        _delta_min_is_defined = false;
        _delta_min_is_complete = false;
        return;
    }
    
    // Test that given delta_min is valid
    if ( delta_min.size() != _n )
        throw NOMAD::Exception ( "OrthogonalMesh.cpp" , __LINE__ ,
                                "set_min_mesh_sizes() delta_min has dimension different than mesh dimension" );
    
    if ( ! delta_min.is_complete() )
        throw NOMAD::Exception (  "OrthogonalMesh.hpp" , __LINE__ ,
                                "set_min_mesh_sizes(): delta_min has some defined and undefined values" );
    
    _delta_min.reset(_n);
    _delta_min_is_defined = true;
    _delta_min_is_complete = true;
    _delta_min=delta_min;
    
    std::string error;
    for ( int k = 0 ; k < _n ; ++k )
    {
        
        // we check that Delta_min <= Delta_0 and that delta_min <= delta_0:
        if ( delta_min[k].is_defined() && _delta_0[k] < delta_min[k] )
            _delta_min[k]=_delta_0[k];
        
        if ( delta_min[k].is_defined() && _Delta_0[k] < delta_min[k] )
            _delta_min[k]=_Delta_0[k];
    }
    
    if ( !error.empty() )
        throw NOMAD::Exception ( "OrthogonalMesh.cpp" , __LINE__ , error );
    
    
}


/// Manually set the min poll size per coordinate.
void NOMAD::OrthogonalMesh::set_min_poll_sizes ( const NOMAD::Point & Delta_min )
{
    
    // If Delta_min undefined than _Delta_min->undefined
    if ( ! Delta_min.is_defined() )
    {
        _Delta_min.clear();
        _Delta_min_is_defined = false;
        _Delta_min_is_complete = false;
        return;
    }
    
    // Test that given Delta_min is valid
    if ( Delta_min.size() != _n )
        throw NOMAD::Exception ( "OrthogonalMesh.cpp" , __LINE__ ,
                                "set_min_poll_sizes() Delta_min has dimension different than mesh dimension" );
    
    // Test that the given Delta_min is complete
    if ( ! Delta_min.is_complete() )
        throw NOMAD::Exception (  "OrthogonalMesh.hpp" , __LINE__ ,
                                "set_min_poll_sizes(): Delta_min has some defined and undefined values" );
    
    _Delta_min.reset( _n );
    _Delta_min = Delta_min;
    _Delta_min_is_defined = true;
    _Delta_min_is_complete = true;
    
    std::string error;
    for ( int k = 0 ; k < _n ; ++k )
    {
        // we check that Delta_min <= Delta_0 :
        if ( Delta_min[k].is_defined() && _Delta_0[k] < Delta_min[k] )
            _Delta_min[k]=_Delta_0[k];
    }
    
    if ( !error.empty() )
        throw NOMAD::Exception ( "OrthogonalMesh.cpp" , __LINE__ , error );
    
}



/*-----------------------------------------------------------*/
/*             set delta_0                                   */
/*-----------------------------------------------------------*/
void NOMAD::OrthogonalMesh::set_delta_0 ( const NOMAD::Point & d )
{
    
    if ( d.size() != _delta_0.size() )
        throw NOMAD::Exception ( "OrthogonalMesh.hpp" , __LINE__ ,
                                "NOMAD::OrthogonalMesh::set_delta_0(): dimension of provided delta_0 must be consistent with their previous dimension" );
    
    _delta_0=d;
}

/*-----------------------------------------------------------*/
/*             set Delta_0                                   */
/*-----------------------------------------------------------*/
void NOMAD::OrthogonalMesh::set_Delta_0 ( const NOMAD::Point & d )
{
    
    if ( d.size() != _Delta_0.size() )
        throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                "NOMAD::XMesh::set_Delta_0(): dimension of provided Delta_0 must be consistent with their previous dimension" );
    
    _Delta_0=d;
}
