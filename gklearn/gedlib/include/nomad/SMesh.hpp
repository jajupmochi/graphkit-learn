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
 \file   SMesh.hpp
 \brief  Class for the static orthogonal MADS mesh (headers)
 \author Christophe Tribes
 \date   2010-04-06
 \see    OrthogonalMesh.cpp SMesh.cpp
 */
#ifndef __SMESH__
#define __SMESH__

#include "Point.hpp"
#include "OrthogonalMesh.hpp"

namespace NOMAD {
    
    /// Class for the MADS orthogonal static mesh.
    /**
     - The static mesh is defined with the basic directions and a
     mesh size parameter delta^k.
     - The mesh size parameter is defined with a single mesh index (the integer r^k)
     and the initial mesh size delta^0 for all coordinates.
     - The poll size parameter Delta^k (single value for all coordinates) is not
     used to define the mesh but to define the poll trial points.
     - At each MADS iteration the mesh is updated with
     delta^k+1 = tau^w+ delta^k and w+ >= 0 (dominating iteration)
     or with
     delta^k+1 = tau^w- delta^k and w- < 0 (iteration failure).
     The mesh is not changed after improving iterations.
     - Mesh and poll size parameters are stored as NOMAD::Point objects
     (one value for each variable). The scaling is done once based on initial mesh size.
     - See the MADS papers for more details on the mesh.
     */
	
	class SMesh : public NOMAD::OrthogonalMesh {
		
	private:
		
		int _initial_mesh_index;
		int	_mesh_index;
		int	_min_mesh_index;           ///< Minimal value reached by the mesh index
		int	_max_mesh_index;           ///< Maximal value reached by the mesh index
		
		/*--------------------------------------------------------------*/
		
		/// Private affectation operator.
		/**
		 \param m The right-hand side object -- \b IN.
		 */
		const SMesh & operator = ( const SMesh & m );
		
		/// Check the minimal poll size criterion.
		bool check_min_poll_size_criterion ( ) const;
		
		/// Check the minimal mesh size criterion.
		bool check_min_mesh_size_criterion ( ) const;
        
        /// Access to the mesh size parameter delta^k.
        /**
         \return delta    The mesh size parameter delta^k -- \b OUT.
         \param  i        The index of the mesh size
         */
        NOMAD::Double get_delta ( int i ) const ;
        
        
        /// Access to the poll size parameter Delta^k.
        /**
         \return Delta    The poll size parameter Delta^k -- \b OUT.
         \param  i        The index of the poll size
         */
        NOMAD::Double get_Delta ( int i ) const ;
        

		
		/*--------------------------------------------------------------*/
		
	public:
		
		/// Constructor.
		/**
		 \param delta_0					Initial mesh size delta_0                      -- \b IN.
		 \param Delta_min				Minimal poll size Delta_min (may be undefined) -- \b IN.
		 \param delta_min				Minimal mesh size delta_min (may be undefined) -- \b IN.
         \param fixed_variables         Fixed variables                                -- \b IN.
		 \param mesh_update_basis		Mesh update basis (tau), default=4             -- \b IN.
		 \param mesh_coarsening_step	Mesh coarsening step (w+), default=1		   -- \b IN.
		 \param mesh_refining_step		Mesh refining step (w-), default=-1			   -- \b IN.
		 \param initial_mesh_index		Initial mesh index ell_0, default=0			   -- \b IN.
		 \param limit_max_mesh_index	Limit max of the mesh index, default=L_LIMITS  -- \b IN.
		 */
		SMesh ( const NOMAD::Point & delta_0   ,
			   const NOMAD::Point & Delta_min ,
			   const NOMAD::Point & delta_min ,
               const NOMAD::Point & fixed_variables ,
			   NOMAD::Double		mesh_update_basis=4.0,
			   int					mesh_coarsening_step=1,
			   int					mesh_refining_step=-1,
			   int					initial_mesh_index=0,
               int                  limit_max_mesh_index=NOMAD::L_LIMITS)
		: NOMAD::OrthogonalMesh ( false,
                                 delta_0,
								 Delta_min,
								 delta_min,
                                 fixed_variables,
                                 NOMAD::Point(),
								 mesh_update_basis,
								 mesh_coarsening_step,
								 mesh_refining_step ,
                                 limit_max_mesh_index ),
        _initial_mesh_index ( initial_mesh_index ),
        _mesh_index ( _initial_mesh_index ),
        _min_mesh_index ( initial_mesh_index ),
        _max_mesh_index ( initial_mesh_index )  {}
        
        
		/// Copy constructor.
		/**
		 \param m The copied object -- \b IN.
		 */
		SMesh ( const SMesh & m )
		: OrthogonalMesh(m) ,
		_initial_mesh_index ( m._initial_mesh_index ),
		_mesh_index ( m._initial_mesh_index ),
		_min_mesh_index( m._initial_mesh_index ),
		_max_mesh_index ( m._initial_mesh_index ) {}
		
		/// Destructor.
		virtual ~SMesh ( void )
        {
            _delta_0.clear();
            _Delta_0.clear();
            _delta_min.clear();
            _Delta_min.clear();
        }
        
		
		/// Access to the mesh index.
		/**
		 \return A Point with the mesh index.
		 */
		const NOMAD::Point get_mesh_indices ( void  ) const
        {
            return NOMAD::Point( 1 , NOMAD::Double(_mesh_index) );
        }
        
		/// Access to the min mesh index reached so far.
		/**
		 \return A Point with the mesh index.
		 */
		const NOMAD::Point get_min_mesh_indices ( void  ) const
        {
            return NOMAD::Point( 1 , NOMAD::Double(_min_mesh_index) );
        }
		
        
		/// Access to the max mesh index reached so far.
		/**
		 \return A Point with the mesh index.
		 */
		const NOMAD::Point get_max_mesh_indices ( void  ) const
        {
            return NOMAD::Point( 1 , NOMAD::Double(_max_mesh_index) );
        }
		
		
		/// Manually set the mesh index using a point. (set_mesh_indices for consistency with XMesh)
		/**
		 \param r   The mesh index provided as a point -- \b IN.
		 */
		void set_mesh_indices ( const NOMAD::Point & r );
        
		
        
        /// Manually set the limit mesh index used for termination criterion (max value for SMesh).
		/**
		 \param l   The limit mesh index for all coordinates -- \b IN.
		 */
		void set_limit_mesh_index ( int l );
        
        
        
		/// Test if finest mesh so far.
		/**
		 \return True if mesh index greater or equal to the maximal mesh index; False otherwise.
		 */
		bool is_finest(void) const {return _mesh_index >= _max_mesh_index;  }
        
		
		/// Access to the mesh ratios after a success
		/**
		 \return A point with the ratio for each coordinate
		 */
		NOMAD::Point get_mesh_ratio_if_success( void ) const;
		
        
		/// Update the provided mesh indices (the Mesh is unchanged).
		/**
		 \param success			Type of success of the iteration				-- \b IN.
		 \param mesh_indices	The mesh indices before and after the update	-- \b IN/OUT.
		 \param dir				The direction that is considered (opt)			-- \b IN.
		 */
		void update ( NOMAD::success_type success , NOMAD::Point & mesh_indices, const NOMAD::Direction *dir=NULL ) const ;
		
        
		/// Update the Mesh.
		/**
		 \param success			Type of success of the iteration				-- \b IN.
		 \param dir				The direction that is considered (opt)			-- \b IN.
		 */
		void update ( NOMAD::success_type success , const NOMAD::Direction *dir=NULL);
        
		
		/// Reset the mesh to its original size (mesh indices).
		void reset ( void )
        {
            set_mesh_indices( NOMAD::Point(1,NOMAD::Double(_initial_mesh_index))) ;
			_min_mesh_index=_initial_mesh_index ;
			_max_mesh_index=_initial_mesh_index;
        }
		
		/// Access to the mesh size parameter delta^k.
		/**
		 \param delta    The mesh size parameter delta^k -- \b OUT.
		 \return A boolean equal to \c true if all values are
		 strictly inferior than the associated minimal
		 mesh size delta_min
		 (stopping criterion MIN_MESH_SIZE).
		 */
		virtual bool get_delta ( NOMAD::Point & delta ) const ;


        
        
		/// Access to the larget mesh size so far.
		/**
		 \return delta_max    The largest mesh size reached so far -- \b OUT.
		 */
        NOMAD::Point get_delta_max ( void ) const ;
		
		/// Access to the poll size parameter Delta^k.
		/**
		 \param Delta    The poll size parameter Delta^k -- \b OUT.
		 \return A boolean equal to \c true if all values are
		 strictly inferior than the associated minimal
		 mesh size Delta_min
		 (stopping criterion MIN_POLL_SIZE).
		 */
		virtual bool get_Delta ( NOMAD::Point & Delta ) const ;

        
		/// Check the stopping conditions on the minimal poll and mesh sizes.
		/**
		 \param stop           Stop flag                  -- \b IN/OUT.
		 \param stop_reason    Stop reason                -- \b OUT.
		 */
		void check_min_mesh_sizes (	bool             & stop           ,
                                   NOMAD::stop_type & stop_reason      ) const;
		/// Display.
		/**
		 \param out The NOMAD::Display object -- \b IN.
		 */
		void display ( const NOMAD::Display & out ) const;
		
		
        /// Scale and project the ith component of a vector on the mesh
        /**
         \param i	      The vector component number           -- \b IN.
         \param l	      The vector component value            -- \b IN.
         \param round_up  The flag for round up                 -- \b IN.
         \return	      The ith component of a vector after mesh scaling and projection
         */
		NOMAD::Double scale_and_project(int i, const NOMAD::Double & l, bool round_up ) const ;
        
		
	};
}

#endif
