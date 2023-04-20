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
 \file   RNG.hpp
 \brief  Custom class for random number generator
 \author Christophe Tribes and Sebastien Le Digabel 
 \date   2011-09-28
 \see    RNG.cpp
 */

#ifndef __RNG__
#define __RNG__

#include "defines.hpp"
#include "Exception.hpp"

using namespace std;

namespace NOMAD {

    
    /// Class for random number generator
	/**
     This class is used to set a seed for the random number generator and
     get a random integer or a random double between two values.
	 */
	class RNG {
		
	public:
        
        
 		/// Get current seed
		/*
		 /return An integer in [0,UINT32_MAX].
		 */
        static int get_seed ( void )
        {
            return static_cast<int>(_s);
        }
        
		/// Set seed
		/*
		 /param s The seed -- \b IN.
		 */
		static void set_seed(int s);
    
		
		/// Get a random integer as uint32
		/** This function serves to obtain a random number \c
		 /return An integer in the interval [0,UINT32_MAX].
		 */
		static uint32_t rand();
        
		
		/// Get a random number having a normal distribution as double
		/*
         /param a Lower bound  -- \b IN.
         /param b Upper bound  -- \b IN.
         /return A double in the interval [a,b].
		 */
		static double rand(double a, double b)
        {
            return a+((b-a)*NOMAD::RNG::rand())/UINT32_MAX;
        }
		
		/// Get a random number approaching a normal distribution (N(0,Var)) as double
		//  A series of Nsample random numbers Xi in the interval [-sqrt(3*Var);+sqrt(3*Var)] is used -> E[Xi]=0, Var(Xi)=var
		// see http://en.wikipedia.org/wiki/Central_limit_theorem
		/*
         /param Nsample	Number of samples for averaging				-- \b IN.
         /param Var		Variance of the target normal distribution	-- \b IN.
         /return A double in the interval [-sqrt(3*Var);+sqrt(3*Var)].
		 */
		static double normal_rand_mean_0( double Var=1 , int Nsample=12 ) ;
        
        
        /// Get a random number approaching a normal distribution ( N(Mean,Var) ) as double
		/*
         /param Mean	Mean of the target normal distribution		-- \b IN.
         /param Var		Variance of the target normal distribution	-- \b IN.
         /return A random number.
		 */
        static double normal_rand( double Mean=0 , double Var=1 ) ;
        
        /// Reset seed to its default value
        static void reset_private_seed_to_default ( void )
        {
            _x=x_def;
            _y=y_def;
            _z=z_def;
        }

        
        
	private:
        
        static uint32_t x_def,y_def,z_def,_x,_y,_z;  ///< Default parameter value for the random number generator (_s used as the seed).
        
        static int _s;
        

	};
}


#endif
