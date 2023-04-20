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
 \file   SMesh.cpp
 \brief  Class for the MADS mesh (implementation)
 \author Sebastien Le Digabel
 \date   2010-04-06
 \see    SMesh.hpp
 */
#include "SMesh.hpp"



/*-----------------------------------------------------------*/
/*                    update the mesh                        */
/*-----------------------------------------------------------*/
void NOMAD::SMesh::update ( NOMAD::success_type success , const NOMAD::Direction *dir)
{
    // defaults:
    //  full success: lk = lk - 1
    //  failure     : lk = lk + 1
    
    
    if ( success == NOMAD::FULL_SUCCESS )
    {
        _mesh_index -= _coarsening_step;
        if ( _mesh_index < -NOMAD::L_LIMITS )
            _mesh_index = -NOMAD::L_LIMITS;
    }
    else if ( success == NOMAD::UNSUCCESSFUL )
        _mesh_index -= _refining_step;
    
    if ( _mesh_index > _max_mesh_index )
        _max_mesh_index = _mesh_index;
    
    
    if ( _mesh_index < _min_mesh_index )
        _min_mesh_index = _mesh_index;
}

/*-----------------------------------------------------------*/
/* Update the provided mesh indices (the Mesh is unchanged). */
/*-----------------------------------------------------------*/
void NOMAD::SMesh::update ( NOMAD::success_type success ,
                           NOMAD::Point & mesh_indices,
                           const NOMAD::Direction *dir ) const
{
    
    if ( mesh_indices.is_defined() )
    {
        for (int i=0; i < mesh_indices.size() ; i++)
        {
            if ( success == NOMAD::FULL_SUCCESS )
            {
                mesh_indices[i] -= _coarsening_step;
                if ( mesh_indices[i] < -NOMAD::L_LIMITS )
                    mesh_indices[i] = -NOMAD::L_LIMITS;
            }
            else if ( success == NOMAD::UNSUCCESSFUL )
                mesh_indices[i] -= _refining_step;
        }
    }
}


/*-----------------------------------------------------------*/
/*              manually set the mesh index                  */
/*-----------------------------------------------------------*/
void NOMAD::SMesh::set_mesh_indices ( const NOMAD::Point & r )
{
    if (!r.is_defined())
        _mesh_index=0;
    else
        _mesh_index=r[0].NOMAD::Double::round(); // Warning no need to check that the double will convert nicely to an int because mesh_index will range [-50;50]
    
    if ( _mesh_index > _max_mesh_index )
        _max_mesh_index = _mesh_index;
    if ( _mesh_index < _min_mesh_index )
        _min_mesh_index = _mesh_index;
}



/*-----------------------------------------------------------*/
/*             set the limit mesh index (max value)          */
/*-----------------------------------------------------------*/
void NOMAD::SMesh::set_limit_mesh_index ( int l )
{
    _limit_mesh_index=l;
}



/*-----------------------------------------------------------*/
/*                           display                         */
/*-----------------------------------------------------------*/
void NOMAD::SMesh::display ( const NOMAD::Display & out ) const
{
    out << "n                       : " << get_n()                   << std::endl
    << "mesh update basis       : " << _update_basis        << std::endl
    << "mesh coarsening step: " << _coarsening_step << std::endl
    << "mesh refining step  : " << _refining_step   << std::endl
    << "initial mesh size       : "
    << "(" << _delta_0 << " )" << std::endl;
    out << "minimal mesh size       : ";
    if ( _delta_min.is_defined() )
        out << "(" << _delta_min << " )" << std::endl;
    else
        out << "none";
    out << std::endl
    << "minimal poll size       : ";
    if ( _Delta_min_is_defined )
        out << "(" << _Delta_min << " )" << std::endl;
    else
        out << "none";
    out << std::endl;
}

/*----------------------------------------------------------*/
/*  check the stopping conditions on the minimal poll size  */
/*  and on the minimal mesh size                            */
/*----------------------------------------------------------*/
void NOMAD::SMesh::check_min_mesh_sizes ( bool             & stop           ,
                                         NOMAD::stop_type & stop_reason      ) const
{
    if ( stop )
        return;
    
    // 1. mesh index tests:
    if ( abs ( _mesh_index ) > NOMAD::L_LIMITS )
    {
        stop        = true;
        stop_reason = NOMAD::L_LIMITS_REACHED;
    }
    
    // 2. delta_k^p (poll size) tests:
    if ( check_min_poll_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_P_MIN_REACHED;
    }
    
    // 3. delta_k^m (mesh size) tests:
    if ( check_min_mesh_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_M_MIN_REACHED;
    }
}

/*-----------------------------------------------------------*/
/*              check the minimal poll size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::SMesh::check_min_poll_size_criterion ( ) const
{
    if ( !_Delta_min_is_defined )
        return false;
    NOMAD::Point Delta;
    return get_Delta ( Delta  );
}

/*-----------------------------------------------------------*/
/*              check the minimal mesh size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::SMesh::check_min_mesh_size_criterion ( ) const
{
    if ( !_delta_min.is_defined() )
        return false;
    NOMAD::Point delta;
    return get_delta ( delta );
}

/*----------------------------------------------------------------*/
/*  get delta (mesh size parameter)                               */
/*       delta^k = delta^0 \tau^{ell_0^+ - ell_k^+}               */
/*----------------------------------------------------------------*/
/*  the function also returns true if one value is < delta_min    */
/*  (stopping criterion MIN_MESH_SIZE)                            */
/*----------------------------------------------------------------*/
bool NOMAD::SMesh::get_delta ( NOMAD::Point & delta ) const
{
    delta.reset ( _n );
    
    bool delta_min_is_defined=_delta_min.is_defined();
    
    // power_of_tau = tau^{ max{0,l0} - max{0,lk} }:
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() ,
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index          > 0) ? _mesh_index          : 0)   );
    
    bool stop    = false;
    
    // delta^k = power_of_tau * delta^0:
    for ( int i = 0 ; i < _n ; ++i )
    {
        delta[i] = _delta_0[i] * power_of_tau;
        if ( !stop && delta_min_is_defined && delta[i] < _delta_min[i] )
            stop = true;
        
    }
    
    return stop;
}

/*----------------------------------------------------------------*/
/*  get delta (mesh size parameter)                               */
/*       delta^k = delta^0 \tau^{ell_0^+ - ell_k^+}               */
/*----------------------------------------------------------------*/
NOMAD::Double NOMAD::SMesh::get_delta ( int i ) const
{
    
    
    // power_of_tau = tau^{ max{0,l0} - max{0,lk} }:
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() ,
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index          > 0) ? _mesh_index          : 0)   );
    
    
    NOMAD::Double delta = _delta_0[i] * power_of_tau;
    
    return delta;
}


/*----------------------------------------------------------------*/
/*  get delta_max (the larget mesh size)                          */
/*----------------------------------------------------------------*/
NOMAD::Point NOMAD::SMesh::get_delta_max ( ) const
{
    
    NOMAD::Point delta_max ( _n );
    
    // power_of_tau = tau^{ max{0,l0} - max{0,lk} }:
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() ,
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_min_mesh_index          > 0) ? _min_mesh_index          : 0)   );
    
    // delta^k = power_of_tau * delta^0:
    for ( int i = 0 ; i < _n ; ++i )
        delta_max[i] = _delta_0[i] * power_of_tau;
    
    return delta_max;
}


/*-------------------------------------------------------------------*/
/*  get Delta (poll size parameter)                                  */
/*       Delta^k = Delta^m_k \tau^{ |ell_k|/2 }                      */
/*                 = delta^0 \tau^{ell_0^+ - ell_k^+ + |ell_k|/2}    */
/*-------------------------------------------------------------------*/
/*  the function also returns true if all values are < Delta_min     */
/*  (stopping criterion MIN_POLL_SIZE)                               */
/*-------------------------------------------------------------------*/
bool NOMAD::SMesh::get_Delta ( NOMAD::Point & Delta ) const
{
    
    Delta.reset ( _n );
    
    // power_of_tau = tau^{ max{0,l0} - max{0,lk} + |lk|/2}:
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() , abs(_mesh_index) / 2.0             +
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index          > 0) ? _mesh_index          : 0)   );
    
    bool stop    = true;
    
    // Delta^k = power_of_tau * Delta^0:
    for ( int i = 0 ; i < _n ; ++i )
    {
        Delta[i] = _Delta_0[i] * power_of_tau;
        if ( !_Delta_min_is_complete || Delta[i] >= _Delta_min[i] )
            stop = false;
        
        if ( _Delta_min_is_defined && _Delta_min[i].is_defined() && Delta[i] < _Delta_min[i] )
            Delta[i]=_Delta_min[i];
    }
    
    return stop;
}

/*-------------------------------------------------------------------*/
/*  get Delta (poll size parameter)                                  */
/*       Delta^k = Delta^m_k \tau^{ |ell_k|/2 }                      */
/*                 = delta^0 \tau^{ell_0^+ - ell_k^+ + |ell_k|/2}    */
/*-------------------------------------------------------------------*/
/*  the function returns Delta >= Delta_min                          */
/*-------------------------------------------------------------------*/
NOMAD::Double NOMAD::SMesh::get_Delta ( int i ) const
{
    
    // power_of_tau = tau^{ max{0,l0} - max{0,lk} + |lk|/2}:
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() , abs(_mesh_index) / 2.0             +
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index          > 0) ? _mesh_index          : 0)   );
    
    NOMAD::Double Delta = _Delta_0[i] * power_of_tau;
    
    if ( _Delta_min_is_defined && _Delta_min[i].is_defined() && Delta < _Delta_min[i] )
        Delta=_Delta_min[i];
    
    return Delta;
}


NOMAD::Double NOMAD::SMesh::scale_and_project(int i, const NOMAD::Double & l, bool round_up ) const
{
    NOMAD::Double delta = get_delta ( i );
    NOMAD::Double Delta = get_Delta ( i );
    
    
    if ( delta.is_defined() && Delta.is_defined() && i <= _n)
    {
        NOMAD::Double d= Delta / delta * l;
        if ( ! round_up )
            return d.NOMAD::Double::round()*delta;
        else
            return d.NOMAD::Double::ceil()*delta;
    }
    else
        throw NOMAD::Exception ( "SMesh.cpp" , __LINE__ ,
                                "Mesh scaling and projection cannot be performed!" );
    
}




NOMAD::Point NOMAD::SMesh::get_mesh_ratio_if_success ( void ) const
{
    
    NOMAD::Double power_of_tau
    = pow ( _update_basis.value() ,
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index          > 0) ? _mesh_index          : 0)   );
    
    NOMAD::Double power_of_tau_if_success
    = pow ( _update_basis.value() ,
           ( (_initial_mesh_index > 0) ? _initial_mesh_index : 0) -
           ( (_mesh_index - _coarsening_step          > 0) ? _mesh_index - _coarsening_step : 0)   );
    
    try
    {
        NOMAD::Double ratio_scalaire = power_of_tau_if_success/power_of_tau;
        return NOMAD::Point( _n , ratio_scalaire );
    }
    catch ( NOMAD::Double::Invalid_Value & )
    {
        return NOMAD::Point( _n,-1 );
    }
}
