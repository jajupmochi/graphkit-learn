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
 \file   XMesh.cpp
 \brief  Class for the MADS xmesh (implementation)
 \author Christophe Tribes
 \date   2014-07
 \see    XMesh.hpp
 */
#include "XMesh.hpp"
#include <math.h>


/*-----------------------------------------------------------*/
/*                    init the XMesh                         */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::init ( )
{
    
    if ( _limit_mesh_index >0 )
        throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                "NOMAD::XMesh::XMesh(): limit mesh index must be <=0 " );
    
    // The delta_0 depends on Delta_0 and the problem dimension
    _delta_0=_Delta_0;
    _delta_0*=pow(_n_free_variables,-0.5);
    

    _r.resize( _n );
    _r_max.resize( _n );
    _r_min.resize( _n );
    
    for ( int k = 0 ; k < _n ; ++k )
    {
        _r[k]=0;
        _r_max[k]=0;
        _r_min[k]=0;
    }
    
    
}


/*-----------------------------------------------------------*/
/* Update the provided mesh indices (the Mesh is unchanged). */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::update ( NOMAD::success_type success ,
                           NOMAD::Point & mesh_indices,
                           const NOMAD::Direction *dir ) const
{
    
    if ( mesh_indices.is_defined() )
    {
        
        if ( dir && dir->size() != mesh_indices.size() )
            throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                    "NOMAD::XMesh::update(): mesh_indices and dir have different sizes" );
        
        for (int i=0; i < mesh_indices.size() ; i++)
        {
            if ( success == NOMAD::FULL_SUCCESS )
            {
                
                if ( (*dir)[i] !=0.0)
                    mesh_indices[i] += _coarsening_step;
                
                if ( mesh_indices[i] > -NOMAD::XL_LIMITS )
                    mesh_indices[i] = -NOMAD::XL_LIMITS;
            }
            else if ( success == NOMAD::UNSUCCESSFUL )
                mesh_indices[i] += _refining_step;
        }
    }
}



/*-----------------------------------------------------------*/
/*                    update the XMesh                       */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::update ( NOMAD::success_type success , const NOMAD::Direction * dir)
{
    // defaults:
    //  full success: r^k_j = r^k_j + 1 if (dir_k != 0 and anistropic_mesh) and r^k_j remains the same if dir_k=0
    //  failure     : r^k_j = r^k_j - 1
    
    // Granularity is only used to block Delta from decreasing but mesh indices are updated
    
    
    if ( dir && dir->size() != _n )
        throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                "NOMAD::XMesh::update(): delta_0 and dir have different sizes" );
    
    
    if ( success == NOMAD::FULL_SUCCESS )
    {
        // Evaluate ||d||_inf
        NOMAD::Double norm_inf=0;
        NOMAD::Point delta=NOMAD::OrthogonalMesh::get_delta();
        
        if ( _anisotropic_mesh )
        {
            for (int i=0; i < _n; i++)
            {
                if ( dir && (*dir)[i].abs()/delta[i] > norm_inf )
                    norm_inf=(*dir)[i].abs()/delta[i];
            }
        }
        else
            norm_inf=-1;
        
        // Determine current max mesh index
        NOMAD::Double max_index=NOMAD::XL_LIMITS;
        for (int j=0; j < _n; j++)
            if ( _r[j] > max_index )
                max_index=_r[j];
        
        
        // Update mesh indices for coordinates with |dir_j|>1/n_free_variables ||dir||_inf or mesh index >=-2
        for (int i=0; i < _n; i++)
        {
            if ( ! dir || (*dir)[i].abs()/delta[i] > norm_inf/_n_free_variables || _r[i] >=-2 )
            {
                _r[i] += _coarsening_step;
                
                if (_r[i]  > -NOMAD::XL_LIMITS )
                    _r[i] = -NOMAD::XL_LIMITS;
                
                if ( _r[i] > _r_max[i] )
                    _r_max[i] = _r[i];
                
            }
        }
        
        
        // Udate mesh indices for coordinates with |dir_l| < 1/n_free_variables ||dir||_inf and mesh index < 2*max_mesh_index
        for (int l=0; l < _n; l++)
        {
            if ( dir &&  _r[l] < -2 && (*dir)[l].abs()/delta[l] <= norm_inf/_n_free_variables && _r[l] < 2*max_index )
                _r[l]+= _coarsening_step;
        }
        
    }
    else if ( success == NOMAD::UNSUCCESSFUL )
        for (int i=0; i< _n; i++)
        {
            _r[i] += _refining_step;
            
            if ( _r[i] < _r_min[i] )
                _r_min[i] = _r[i];
            
        }
}


/*-----------------------------------------------------------*/
/*                           display                         */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::display ( const NOMAD::Display & out ) const
{
    out << "n                       : " << _n               << std::endl
    << "tau                        : " << _update_basis    << std::endl
    << "poll coarsening exponent: " << _coarsening_step << std::endl
    << "poll refining exponent  : " << _refining_step   << std::endl;
    out << "minimal mesh size       : ";
    if ( _delta_min.is_defined() )
        out << "(" << _delta_min     << " )" << std::endl;
    else
        out << "none";
    out << std::endl
    << "minimal poll size       : ";
    if ( _Delta_min_is_defined )
        out << "( " << _Delta_min     << " )" << std::endl;
    else
        out << "none";
    
    out << std::endl << "initial poll size       : ";
    if (_Delta_0.is_defined())
        out <<"( " << _Delta_0     << " )" << std::endl;
    else
        out <<"( none )" << std::endl;
    
    out << std::endl << "initial mesh size       : ";
    
    if ( _delta_0.is_defined() )
        out <<"( " << _delta_0     << " )" << std::endl;
    else
        out <<"( none )" << std::endl;
    
    out << std::endl;
}


/*----------------------------------------------------------*/
/*  check the stopping conditions on the minimal poll size  */
/*  and on the minimal mesh size                            */
/*----------------------------------------------------------*/
void NOMAD::XMesh::check_min_mesh_sizes ( bool             & stop           ,
                                         NOMAD::stop_type & stop_reason      ) const
{
    if ( stop )
        return;
    
    // Coarse mesh stopping criterion
    stop=false;
    
    for (int i=0;i<_n;i++)
        if ( _r[i] > -NOMAD::XL_LIMITS )
        {
            stop        = true;
            break;
        }
    
    if (stop)
    {
        stop_reason = NOMAD::XL_LIMITS_REACHED;
        return;
    }
    
    stop=true;
    // Fine mesh stopping criterion
    // All mesh indices must < _limit_mesh_index to trigger this stopping criterion
    for (int i=0;i<_n;i++)
    {
        if ( _r[i] >= _limit_mesh_index )
        {
            stop        = false;
            break;
        }
    }
    if (stop)
    {
        stop_reason = NOMAD::XL_LIMITS_REACHED;
        return;
    }
    
    // 2. Delta^k (poll size) tests:
    if ( check_min_poll_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_P_MIN_REACHED;
    }
    
    // 3. delta^k (mesh size) tests:
    if ( check_min_mesh_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_M_MIN_REACHED;
    }
}

/*-----------------------------------------------------------*/
/*              check the minimal poll size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::XMesh::check_min_poll_size_criterion ( ) const
{
    if ( !_Delta_min_is_defined )
        return false;
    
    NOMAD::Point Delta;
    return get_Delta ( Delta );
}

/*-----------------------------------------------------------*/
/*              check the minimal mesh size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::XMesh::check_min_mesh_size_criterion ( ) const
{
    if ( !_delta_min.is_defined() )
        return false;
    
    NOMAD::Point delta;
    return get_delta ( delta );
}


/*--------------------------------------------------------------*/
/*  get delta (mesh size parameter)                             */
/*       delta^k = delta^0 \tau ^min{0,2*r^k}                   */
/*--------------------------------------------------------------*/
/*  the function also returns true if ALL delta[i] < delta_min  */
/*--------------------------------------------------------------*/
bool NOMAD::XMesh::get_delta ( NOMAD::Point & delta ) const
{
    
    delta.resize(_n);
    
    bool delta_min_is_defined=_delta_min.is_defined();
    bool stop    = true;
    
    // delta^k = power_of_beta * delta^0:
    for ( int i = 0 ; i < _n ; ++i )
    {

        delta[i] = get_delta( i );
        
        if ( stop && delta_min_is_defined && _delta_min[i].is_defined() && delta[i] >= _delta_min[i] )
            stop = false;
    }
    
    return stop;
}


/*--------------------------------------------------------------*/
/*  get delta (mesh size parameter)                             */
/*       delta^k = delta^0 \beta ^min{0,2*r^k}                  */
/*--------------------------------------------------------------*/
NOMAD::Double NOMAD::XMesh::get_delta ( int i ) const
{
    
    // delta^k = power_of_beta * delta^0:
    NOMAD::Double power_of_beta = pow ( _update_basis.value() , ( (_r[i] >= 0) ? 0 : 2*_r[i].value() )  );
    
    return _delta_0[i] * power_of_beta;
    
}



/*---------------------------------------------------------------*/
/*  get Delta (poll size parameter)                              */
/*       Delta^k = Delta^0 \beta ^{r^k}                          */
/*---------------------------------------------------------------*/
/*  the function also returns true if all values are < Delta_min */
/*---------------------------------------------------------------*/
bool NOMAD::XMesh::get_Delta ( NOMAD::Point & Delta ) const
{
    
    bool stop    = true;
    Delta.resize(_n);
    
    // delta^k = power_of_tau * delta^0:
    for ( int i = 0 ; i < _n ; ++i )
    {
        Delta[i] = get_Delta( i );
        
        if (  stop && ! _fixed_variables[i].is_defined() && ( !_Delta_min_is_complete  || Delta[i] >= _Delta_min[i] ) )
            stop = false;
    }
    
    return stop;
}

/*--------------------------------------------------------------*/
/*  get Delta_i  (poll size parameter)                          */
/*       Delta^k = Delta^0 \beta ^{r^k}                         */
/*--------------------------------------------------------------*/
NOMAD::Double NOMAD::XMesh::get_Delta ( int i ) const
{
    NOMAD::Double Delta = _Delta_0[i] * pow( _update_basis.value() , _r[i].value() );

    return Delta;
}


bool NOMAD::XMesh::is_finest ( ) const
{
    for ( int i = 0 ; i < _n ; ++i )
    {
        if ( _r[i] > _r_min[i] )
            return false;
    }
    return true;
}


/*-----------------------------------------------------------*/
/*             set the mesh indices                          */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::set_mesh_indices ( const NOMAD::Point & r )
{
    
    if ( r.size() != _n )
        throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                "NOMAD::XMesh::set_mesh_indices(): dimension of provided mesh indices must be consistent with their previous dimension" );
    
    _r=r;
    for ( int i = 0 ; i < _n ; ++i )
    {
        if ( r[i] > _r_max[i] )
            _r_max[i] = r[i];
        if ( r[i] < _r_min[i] )
            _r_min[i] = r[i];
    }
}


/*-----------------------------------------------------------*/
/*     set the limit mesh index (min value for XMesh)        */
/*-----------------------------------------------------------*/
void NOMAD::XMesh::set_limit_mesh_index ( int l )
{
    _limit_mesh_index=l;
}




/*-----------------------------------------------------------*/
/*              scale and project on the mesh                */
/*-----------------------------------------------------------*/
NOMAD::Double NOMAD::XMesh::scale_and_project(int i, const NOMAD::Double & l, bool round_up) const
{
    NOMAD::Double delta = get_delta( i );
    NOMAD::Double Delta = get_Delta( i );
    
    if ( i<=_n && delta.is_defined() && Delta.is_defined() )
    {

        NOMAD::Double d = Delta / delta * l;
        if ( ! round_up )
            return ( d < 0.0 ? -std::floor(.5-d.value()) : std::floor(.5+d.value()) ) * delta;
        else
            return d.NOMAD::Double::ceil()*delta;
    }
    else
        throw NOMAD::Exception ( "XMesh.cpp" , __LINE__ ,
                                "Mesh scaling and projection cannot be performed!" );
    
    
}


NOMAD::Point NOMAD::XMesh::get_mesh_ratio_if_success ( void ) const
{
    
    try
    {
        NOMAD::Point ratio( _n );
        for (int i=0 ; i < _n ; i++)
        {
            NOMAD::Double power_of_tau
            = pow ( _update_basis.value() , ( (_r[i] >= 0) ? 0 : 2*_r[i].value() )  );
            
            NOMAD::Double power_of_tau_if_success
            = pow ( _update_basis.value() , ( (_r[i] + _coarsening_step >= 0) ? 0 : 2*(_r[i].value() + _coarsening_step) )  );
            
            ratio[i] = power_of_tau_if_success/power_of_tau;
            
        }
        
        return ratio;
    }
    catch ( NOMAD::Double::Invalid_Value & )
    {
        return NOMAD::Point( _n,-1 );
    }
}



