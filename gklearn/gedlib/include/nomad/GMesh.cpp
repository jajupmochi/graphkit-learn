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
 \file   GMesh.cpp
 \brief  Class for the MADS xmesh (implementation)
 \author Christophe Tribes
 \date   2014-07
 \see    GMesh.hpp
 */
#include "GMesh.hpp"
#include <math.h>


/*-----------------------------------------------------------*/
/*                    init the GMesh                       */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::init ( )
{

    if ( _limit_mesh_index > 0 )
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::GMesh(): limit mesh index must be <=0 " );
    // Set the mesh indices
    _r.resize( _n );
    _r_max.resize( _n );
    _r_min.resize( _n );
    
    for ( int k = 0 ; k < _n ; ++k )
    {
        _r[k]=0;
        _r_max[k]=0;
        _r_min[k]=0;
    }
    
    // Set the mesh mantissas and exponents
    init_poll_size_granular ( _Delta_0 );
    
    // Update mesh and poll after granular sizing
    _Delta_0_exp = _Delta_exp ;
    _Delta_0_mant = _Delta_mant;
    get_Delta ( _Delta_0 );
    get_delta ( _delta_0 );
    

    
}


/*-----------------------------------------------------------*/
/* Update the provided mesh indices (the Mesh is unchanged). */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::update ( NOMAD::success_type success ,
                           NOMAD::Point & mesh_indices,
                           const NOMAD::Direction *dir ) const
{
    
    if ( mesh_indices.is_defined() )
    {
        
        if ( dir && dir->size() != mesh_indices.size() )
            throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                    "NOMAD::GMesh::update(): mesh_indices and dir have different sizes" );
        
        for (int i=0; i < mesh_indices.size() ; i++)
        {
            if ( ! _fixed_variables.is_defined() && success == NOMAD::FULL_SUCCESS )
            {
                mesh_indices[i] ++;
                
                if ( mesh_indices[i] > -NOMAD::GL_LIMITS )
                    mesh_indices[i] = -NOMAD::GL_LIMITS;
            }
            else if ( ! _fixed_variables.is_defined() && success == NOMAD::UNSUCCESSFUL )
                mesh_indices[i] --;
        }
    }
}



/*-----------------------------------------------------------*/
/*                    update the granular mesh                        */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::update ( NOMAD::success_type success , const NOMAD::Direction * d )
{
    
    if ( d && d->size() != _n )
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::update(): delta_0 and d have different sizes" );
    
    if ( success == NOMAD::FULL_SUCCESS )
    {
        
        NOMAD::Double min_rho=NOMAD::INF;
        for ( int i=0 ; i < _n ;i++ )
        {
            if ( _granularity[i] == 0 && !_fixed_variables[i].is_defined() )
                min_rho=NOMAD::min ( min_rho , get_rho(i) );
        }
        
        for (int i=0 ; i < _n ; i++ )
        {
            
            // Test for producing anisotropic mesh + correction to prevent mesh collapsing for some variables ( ifnot )
            if ( !_fixed_variables[i].is_defined() )
            {
                

                if ( ! d || ! _anisotropic_mesh ||
                     fabs((*d)[i].value())/get_delta(i).value()/get_rho(i).value() > 0.1 ||
                    ( _granularity[i] == 0  && _Delta_exp[i] < _Delta_0_exp[i] && get_rho(i) > min_rho*min_rho ))
                {
                    // update the mesh index
                    ++_r[i];
                    _r_max[i]=NOMAD::max(_r[i],_r_max[i]);
                    
                    // update the mantissa and exponent
                    if ( _Delta_mant[i] == 1 )
                        _Delta_mant[i]= 2;
                    else if ( _Delta_mant[i] == 2 )
                        _Delta_mant[i]=5;
                    else
                    {
                        _Delta_mant[i]=1;
                        ++_Delta_exp[i];
                    }
                }
            }
        }
    }
    else if ( success == NOMAD::UNSUCCESSFUL )
    {
        for (int i=0 ; i < _n ; i++ )
        {
            if ( !_fixed_variables[i].is_defined() )
            {
                
                // update the mesh index
                --_r[i];
                
                // update the mesh mantissa and exponent
                if ( _Delta_mant[i] == 1 )
                {
                    _Delta_mant[i]= 5;
                    --_Delta_exp[i];
                }
                else if ( _Delta_mant[i] == 2 )
                    _Delta_mant[i]=1;
                else
                    _Delta_mant[i]=2;
                
                
                if ( _granularity[i] > 0 && _Delta_exp[i]==-1 && _Delta_mant[i]==5 )
                {
                    ++_r[i];
                    _Delta_exp[i]=0;
                    _Delta_mant[i]=1;
                }
            }
            // Update the minimal mesh index reached so far
            _r_min[i]=NOMAD::min(_r[i],_r_min[i]);
            
        }
    }
    
    
}


/*-----------------------------------------------------------*/
/*                           display                         */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::display ( const NOMAD::Display & out ) const
{
    
    out << "n                       : " << _n               << std::endl;
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
    if ( _Delta_0.is_defined() )
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
void NOMAD::GMesh::check_min_mesh_sizes ( bool             & stop           ,
                                         NOMAD::stop_type & stop_reason      ) const
{
    if ( stop )
        return;
    
    // Coarse mesh stopping criterion
    stop=false;
    for (int i=0;i<_n;i++)
        if ( _r[i] > -NOMAD::GL_LIMITS )
        {
            stop        = true;
            break;
        }
    
    if ( stop )
    {
        stop_reason = NOMAD::GL_LIMITS_REACHED;
        return;
    }

    stop=true;

    // Fine mesh stopping criterion (do not apply when all variables have granularity
    // All mesh indices must < _limit_mesh_index for all continuous variables (granularity==0) and
    // and mesh size = granularity for all granular variables to trigger this stopping criterion
    if ( _all_granular )
    {
        // Do not stop because of to fine a mesh if all variables are granular
        stop =false;
    }
    else
    {
        for ( int i=0 ; i <_n ; i++ )
        {
            // Do not stop if the mesh size of a variable is strictly larger than its granularity
            if ( _granularity[i] > 0 && ! _fixed_variables[i].is_defined() && get_delta(i) > _granularity[i] )
            {
                stop = false;
                break;
            }
            
            // Do not stop if the mesh of a variable is above the limit mesh index
            if ( _granularity[i] == 0 && ! _fixed_variables[i].is_defined() && _r[i] >= _limit_mesh_index )
            {
                stop = false;
                break;
            }
        }
    }
    
    if ( stop )
    {
        stop_reason = NOMAD::GL_LIMITS_REACHED;
        return;
    }
    
    // 2. Delta^k (poll size) tests:
    if ( check_min_poll_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_P_MIN_REACHED;
        return;
    }
    
    // 3. delta^k (mesh size) tests:
    if ( check_min_mesh_size_criterion ( ) )
    {
        stop        = true;
        stop_reason = NOMAD::DELTA_M_MIN_REACHED;
        return;
    }
}

/*-----------------------------------------------------------*/
/*              check the minimal poll size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::GMesh::check_min_poll_size_criterion ( ) const
{
    if ( !_Delta_min_is_defined )
        return false;
    
    NOMAD::Point Delta;
    return get_Delta ( Delta );
}

/*-----------------------------------------------------------*/
/*              check the minimal mesh size (private)        */
/*-----------------------------------------------------------*/
bool NOMAD::GMesh::check_min_mesh_size_criterion ( ) const
{
    if ( !_delta_min.is_defined() )
        return false;
    
    NOMAD::Point delta;
    return get_delta ( delta );
}


/*--------------------------------------------------------------*/
/*  get delta (mesh size parameter)                                */
/*--------------------------------------------------------------*/
/*  the function also returns true if ALL variables             */
/*  with delta_min verify delta[i] < delta_min[i]               */
/*--------------------------------------------------------------*/
bool NOMAD::GMesh::get_delta ( NOMAD::Point & delta ) const
{
    delta.resize(_n);
    
    bool stop = true;
    
    // delta^k = power_of_beta * delta^0:
    for ( int i = 0 ; i < _n ; ++i )
    {
        
        delta[i] = get_delta( i );
        
        if ( stop && _delta_min_is_defined && ! _fixed_variables[i].is_defined() && _delta_min[i].is_defined() && delta[i] >= _delta_min[i] )
            stop = false;
    }
    
    return stop;
}


/*--------------------------------------------------------------*/
/*  get delta (mesh size parameter)                                */
/*       delta^k = 10^(b^k-|b^k-b_0^k|)                         */
/*--------------------------------------------------------------*/
NOMAD::Double NOMAD::GMesh::get_delta ( int i ) const
{
    
    
    NOMAD::Double delta = pow ( 10 , _Delta_exp[i].value() - std::fabs( _Delta_exp[i].value() - _Delta_0_exp[i].value() ) );
    
    if ( _granularity[i] > 0 )
        delta = _granularity[i] * NOMAD::max( 1.0 , delta );
    
    return delta;
    
}

/*--------------------------------------------------------------*/
/*  get Delta_i  (poll size parameter)                          */
/*       Delta^k = dmin * a^k *10^{b^k}                                 */
/*--------------------------------------------------------------*/
NOMAD::Double NOMAD::GMesh::get_Delta ( int i ) const
{
    
    NOMAD::Double d_min_gran = 1.0;
    
    if ( _granularity[i] > 0 )
        d_min_gran = _granularity[i];
    
    NOMAD::Double Delta = d_min_gran * _Delta_mant[i] * pow ( 10.0, _Delta_exp[i].value() ) ;
    
    return Delta;
}



/*--------------------------------------------------------------*/
/*  get Delta (poll size parameter)                                */
/*--------------------------------------------------------------*/
/*  the function also returns true if all continuous variables   */
/*  have Delta < Delta_min and all granular variables have      */
/*  Delta <= Delta_min                                          */
/*--------------------------------------------------------------*/
bool NOMAD::GMesh::get_Delta ( NOMAD::Point & Delta ) const
{
    
    bool stop    = true;
    Delta.resize(_n);
    
    for ( int i = 0 ; i < _n ; ++i )
    {
        Delta[i] = get_Delta( i );
        
        if (  stop && ! _fixed_variables[i].is_defined() && _granularity[i] == 0 && ( !_Delta_min_is_complete  || Delta[i] >= _Delta_min[i] ) )
            stop = false;
        
        if ( stop && ! _fixed_variables[i].is_defined() && _granularity[i] > 0 && ( !_Delta_min_is_complete  || Delta[i] > _Delta_min[i] ) )
            stop = false;

    }
    
    return stop;
}


/*--------------------------------------------------------------*/
/*  get rho (ratio poll/mesh size)                                */
/*       rho^k = 10^(b^k-|b^k-b_0^k|)                               */
/*--------------------------------------------------------------*/
NOMAD::Double NOMAD::GMesh::get_rho ( int i ) const
{
    NOMAD::Double rho ;
    if ( _granularity[i] > 0  )
        rho = _Delta_mant[i] * min( pow ( 10 , _Delta_exp[i].value() ) , pow ( 10 , std::fabs( _Delta_exp[i].value() - _Delta_0_exp[i].value() ) ) );
    else
        rho = _Delta_mant[i] * pow ( 10 , std::fabs( _Delta_exp[i].value() - _Delta_0_exp[i].value() ) );
    

    return rho;
    
}



bool NOMAD::GMesh::is_finest ( ) const
{
    
    for ( int i = 0 ; i < _n ; ++i )
    {
        if ( ! _fixed_variables[i].is_defined() && _r[i] > _r_min[i] )
            return false;
    }
    return true;
}


/*-----------------------------------------------------------*/
/*             set the mesh indices                          */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::set_mesh_indices ( const NOMAD::Point & r )
{
    if ( r.size() != _n )
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::set_mesh_indices(): dimension of provided mesh indices must be consistent with their previous dimension" );
    
    // Set the mesh indices
    _r=r;
    for ( int i = 0 ; i < _n ; ++i )
    {
        if ( r[i] > _r_max[i] )
            _r_max[i] = r[i];
        if ( r[i] < _r_min[i] )
            _r_min[i] = r[i];
    }
    
    // Set the mesh mantissas and exponents according to the mesh indices
    for ( int i = 0 ; i < _n ; ++i )
    {
        int shift = static_cast<int>( _r[i].value() + _pos_mant_0[i].value() );
        int pos= ( shift + 300 )  % 3 ;
        
        _Delta_exp[i] = std::floor( ( shift + 300.0 )/3.0 ) - 100.0 + _Delta_0_exp[i];
        
        if ( pos == 0 )
            _Delta_mant[i] = 1;
        else if ( pos == 1 )
            _Delta_mant[i] = 2;
        else if ( pos == 2 )
            _Delta_mant[i] = 5;
        else
            throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                    "NOMAD::GMesh::set_mesh_indices(): something is wrong with conversion from index to mantissa and exponent" );
    }
    
}


/*-----------------------------------------------------------*/
/*     set the limit mesh index (min value for GMesh)        */
/*-----------------------------------------------------------*/
void NOMAD::GMesh::set_limit_mesh_index ( int l )
{
    if ( l > 0 )
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::set_limit_mesh_index(): the limit mesh index must be negative or null." );
    _limit_mesh_index=l;
}




/*-----------------------------------------------------------*/
/*              scale and project on the mesh                */
/*-----------------------------------------------------------*/
NOMAD::Double NOMAD::GMesh::scale_and_project(int i, const NOMAD::Double & l, bool round_up) const
{
    
    
    NOMAD::Double delta = get_delta( i );
    
    if ( i <= _n && _Delta_mant.is_defined() && _Delta_exp.is_defined() && delta.is_defined() )
    {
        NOMAD::Double d = get_rho(i) * l;
        return d.roundd() * delta;
    }
    else
        
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::scale_and_project(): mesh scaling and projection cannot be performed!" );
}


NOMAD::Point NOMAD::GMesh::get_mesh_ratio_if_success ( void ) const
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


void NOMAD::GMesh::init_poll_size_granular ( NOMAD::Point & cont_init_poll_size )
{
    
    
    if ( ! cont_init_poll_size.is_defined() || cont_init_poll_size.size() != _n )
        throw NOMAD::Exception ( "GMesh.cpp" , __LINE__ ,
                                "NOMAD::GMesh::init_poll_size_granular(): Inconsistent dimension of the poll size!" );
    
    _Delta_exp.reset ( _n );
    _Delta_mant.reset ( _n );
    _pos_mant_0.reset ( _n );
    
    
    NOMAD::Double d_min;
    
    for ( int i = 0 ; i < _n ; i++ )
    {
        
        if ( _granularity[i].is_defined() && _granularity[i].value() > 0 )
            d_min=_granularity[i];
        else
            d_min=1.0;
        
        int exp= (int)( std::log10( std::fabs( cont_init_poll_size[i].value()/d_min.value() )));
        _Delta_exp[i]=exp;
        
        double cont_mant = cont_init_poll_size[i].value() / d_min.value() * pow ( 10.0 , -exp );
        
        // round to 1, 2 or 5
        if ( cont_mant < 1.5 )
        {
            _Delta_mant[i]=1;
            _pos_mant_0[i]=0;
        }
        else if ( cont_mant >= 1.5 && cont_mant < 3.5 )
        {
            _Delta_mant[i]=2;
            _pos_mant_0[i]=1;
        }
        else
        {
            _Delta_mant[i]=5;
            _pos_mant_0[i]=2;
        }
    }
    
}


bool NOMAD::GMesh::is_finer_than_initial (void) const
{
    
    for (int i =0; i < _n ; ++i )
    {
        if ( !_fixed_variables[i].is_defined() )
        {
            
            // For continuous variables
            if ( _granularity[i]==0 && ( _Delta_exp[i] > _Delta_0_exp[i] || ( _Delta_exp[i] == _Delta_0_exp[i] && _Delta_mant[i] >= _Delta_0_mant[i] ) ) )
                return false;
            
            // For granular variables ( case 1 )
            if ( _granularity[i] > 0 && ( _Delta_exp[i] > _Delta_0_exp[i] || ( _Delta_exp[i] == _Delta_0_exp[i] && _Delta_mant[i] > _Delta_0_mant[i] ) ) )
                return false;
            
            // For granular variables ( case 2 )
            if ( _granularity[i] > 0 && _Delta_exp[i] == _Delta_0_exp[i] && _Delta_mant[i] == _Delta_0_mant[i] && ( _Delta_exp[i] != 0 ||  _Delta_mant[i] != 1 ) )
                return false;
        }
        
        
    }
    
    return true;
}
