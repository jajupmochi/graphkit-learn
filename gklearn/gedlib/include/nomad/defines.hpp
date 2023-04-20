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
 \file   defines.hpp
 \brief  Definitions
 \author Sebastien Le Digabel
 \date   2010-03-23
 */
#ifndef __DEFINES__
#define __DEFINES__

#include <string>
#include <iostream>
#include <sstream>
#include <limits>
#include <limits.h>
#include <cstdlib>


// #define R_VERSION // defined for the R version only

// Matlab version OPTI style (if not defined than GERAD style)
// #define OPTI_VERSION 

// Define in order to display debug information
//#define DEBUG

// define in order to display memory debug information:
//#define MEMORY_DEBUG


#ifdef DEBUG
#ifndef MEMORY_DEBUG
#define MEMORY_DEBUG
#endif
#endif


// CASE Linux using gnu compiler   
#ifdef __gnu_linux__
#define GCC_X
#endif

// CASE OSX using gnu compiler 
#ifdef __APPLE__
#ifdef __GNUC__
#define GCC_X
#endif
#endif

// CASE minGW using gnu compiler
#ifdef __MINGW32__
#define WINDOWS
#ifdef __GNUC__
#define GCC_X
#endif
#endif

// CASE Visual Studio C++ compiler
#ifdef _MSC_VER
#define WINDOWS
#pragma warning(disable:4996)
#endif

// For NOMAD random number generator 
#if !defined(UINT32_MAX)
typedef unsigned int uint32_t;
#define UINT32_MAX	0xffffffff
#endif

// to display model stats for each evaluation at
// which a model has been used (looks better
// with DISPLAY_DEGREE set to zero);
// The displayed stats are:
//   type (1:model_search, 2:model_ordering)
//   mesh_index
//   cardinality of Y
//   width of Y
//   Y condition number
//   h value, model for h, relative error (if constraints)
//   f value, model for f, relative error

// #define MODEL_STATS

namespace NOMAD {
	
	/// Current version:
	const std::string BASE_VERSION = "3.8.1";
	
#ifdef R_VERSION
	const std::string VERSION = BASE_VERSION + ".R";
#else
#ifdef USE_MPI
	const std::string VERSION = BASE_VERSION + ".MPI";
#else
	const std::string VERSION = BASE_VERSION;
#endif
#endif
	
	// Directory separator, plus LGPL and user guide files
#ifdef WINDOWS
	const char        DIR_SEP = '\\';           ///< Directory separator
	const std::string HOME    = "%NOMAD_HOME%"; ///< Home directory
#else
	const char        DIR_SEP = '/';            ///< Directory separator
	const std::string HOME    = "$NOMAD_HOME";  ///< Home directory
#endif
	
	/// Licence file
	const std::string LGPL_FILE = HOME+DIR_SEP+"src"+DIR_SEP+"lgpl.txt";
	
	/// User guide file
	const std::string USER_GUIDE_FILE = HOME+DIR_SEP+"doc"+DIR_SEP+"user_guide.pdf";
	
	/// Examples directory
	const std::string EXAMPLES_DIR = HOME+DIR_SEP+"examples";
	
	/// Tools directory
	const std::string TOOLS_DIR = HOME+DIR_SEP+"tools";
	
	/// Tag for valid cache files
#ifdef GCC_X
	const int CACHE_FILE_ID = 77041301;
#else
#ifdef GCC_WINDOWS
	const int CACHE_FILE_ID = 77041302;
#else
#ifdef _MSC_VER
	const int CACHE_FILE_ID = 77041303;
#else
	const int CACHE_FILE_ID = 77041304;
#endif
#endif
#endif
	
#ifdef USE_MPI
	// MPI constants
	const int   MAX_REQ_WAIT =  3 ;  ///< Maximum time to wait for a request
	const char   STOP_SIGNAL = 'S';  ///< Stop signal
	const char   EVAL_SIGNAL = 'X';  ///< Evaluation signal
	const char  READY_SIGNAL = 'R';  ///< Ready signal
	const char RESULT_SIGNAL = 'O';  ///< Result signal
	const char   WAIT_SIGNAL = 'W';  ///< Wait signal
#endif
	
	/// Maximum number of variables.
	const int MAX_DIMENSION = 1000;
	
	// Old static Mesh index constants
	const int L_LIMITS    = 50;         ///< Limits for the smesh index values
	const int UNDEFINED_L = L_LIMITS+1;  ///< Undefined value for the smesh index

    // xmesh index constants
    const int XL_LIMITS    = -50;         ///< Limits for the xmesh index values
	const int UNDEFINED_XL = XL_LIMITS-1;  ///< Undefined value for the xmesh index
    
    // gmesh index constants
    const int GL_LIMITS    = -50;         ///< Limits for the gmesh index values
    const int UNDEFINED_GL = GL_LIMITS-1;  ///< Undefined value for the gmesh index

   
    
	/// Default epsilon used by NOMAD::Double
	/** Use Parameters::set_EPSILON(), or parameter EPSILON,
	 or NOMAD::Double::set_epsilon() to change it
	 */
	const double DEFAULT_EPSILON = 1e-13;
	
	/// Maximal output value for points used for models.
	const double MODEL_MAX_OUTPUT = 1e10;
	
	/// Default infinity string used by NOMAD::Double
	/** Use Parameters::set_INF_STR(), or parameter INF_STR,
	 or NOMAD::Double::set_inf_str() to change it
	 */
	const std::string DEFAULT_INF_STR = "inf";
	
	/// Default undefined value string used by NOMAD::Double
	/** Use Parameters::set_UNDEF_STR(), or parameter UNDEF_STR,
	 or NOMAD::Double::set_undef_str() to change it
	 */
	const std::string DEFAULT_UNDEF_STR = "NaN";
    
	/// log(10) (for display widths.)
	const double LOG10 = 2.30258509299;
	
    const double NaN = std::numeric_limits<double>::quiet_NaN(); ///< Quiet Not-A-Number
	const double INF = std::numeric_limits<double>::max(); ///< Infinity
    const double P_INF_INT = std::numeric_limits<int>::max(); ///< plus infinity for int
    const double M_INF_INT = std::numeric_limits<int>::min(); ///< minus infinity for int

	
	const double D_INT_MAX = UINT32_MAX; ///< The UINT32_MAX constant as a \c double
	
	// Singular Value Decomposition (SVD) constants:
	const double SVD_EPS      = 1e-13;      ///< Epsilon for SVD
	const int    SVD_MAX_MPN  = 1500;       ///< Matrix maximal size (\c m+n )
	const double SVD_MAX_COND = NOMAD::INF; ///< Max. acceptable cond. number
		
	/// Default value for parameter POINT_DISPLAY_LIMIT
	/** Use Parameters::set_POINT_DISPLAY_LIMIT() or parameter POINT_DISPLAY_LIMIT
	 or Point::set_display_limit() to change it */
	const int DEFAULT_POINT_DISPLAY_LIMIT = 20;
	
	// Display precisions.
	const int DISPLAY_PRECISION_STD = 10;  ///< Standard display precision
	const int DISPLAY_PRECISION_BB  = 15;  ///< Display precision for blackboxes
	
	/// Constant for blackbox files #1.
	const std::string BLACKBOX_INPUT_FILE_PREFIX = "nomad"; 
	
	/// Constant for blackbox files #2.
	const std::string BLACKBOX_INPUT_FILE_EXT = "input";
	
	/// Constant for blackbox files #3.
	const std::string BLACKBOX_OUTPUT_FILE_PREFIX = "nomad";
	
	/// Constant for blackbox files #4.
	const std::string BLACKBOX_OUTPUT_FILE_EXT = "output";
	
	/// Display degree type.
	enum dd_type
    {
		NO_DISPLAY     , ///< No display
		MINIMAL_DISPLAY, ///< Minimal dispay    	
		NORMAL_DISPLAY , ///< Normal display
		FULL_DISPLAY     ///< Full display
    };
	
    /// Types of the variables
	//  (do not modify this order)
	enum bb_input_type
    {
		CONTINUOUS  ,     ///< Continuous variable (default) (R)
		INTEGER     ,     ///< Integer variable              (I)
		CATEGORICAL ,     ///< Categorical variable          (C)
		BINARY            ///< Binary variable               (B)
    };
	
	/// Blackbox outputs type
	enum bb_output_type
    {
		OBJ         ,    ///< Objective value
		EB          ,    ///< Extreme barrier constraint
		PB          ,    ///< progressive barrier constraint
		// PEB           ///< PB constraint that becomes EB once satisfied
		PEB_P       ,    ///< PEB constraint, state P (PB)
		PEB_E       ,    ///< PEB constraint, state E (EB)
		FILTER      ,    ///< Filter constraint
		CNT_EVAL    ,    ///< Output set to 0 or 1 to specify to count or not the
		///<   blackbox evaluation
		STAT_AVG    ,    ///< Stat (average)
		STAT_SUM    ,    ///< Stat (sum)
		UNDEFINED_BBO    ///< Ignored output
    };
	
	/// Formulation for multi-objective optimization
	enum multi_formulation_type
    {
		NORMALIZED            , ///< Normalized formulation
		PRODUCT               , ///< Product formulation
		DIST_L1               , ///< Distance formulation with norm L1
		DIST_L2               , ///< Distance formulation with norm L2
		DIST_LINF             , ///< Distance formulation with norm Linf
		UNDEFINED_FORMULATION   ///< Undefined formulation
    };
	
	/// Poll type
	enum poll_type
    {
		PRIMARY   , ///< Primary poll
		SECONDARY   ///< Secondary poll
    };
	
	/// Poll center feasibility type
	enum poll_center_type
    {
		FEASIBLE                   , ///< Feasible poll center type
		INFEASIBLE                 , ///< Infeasible poll center type
		UNDEFINED_POLL_CENTER_TYPE   ///< Undefined poll center type
    };
	
	/// Search type
	enum search_type
    {
		X0_EVAL          , ///< Starting point evaluation
		POLL             , ///< Poll
		EXTENDED_POLL    , ///< Extended poll
		SEARCH           , ///< Generic search
		CACHE_SEARCH     , ///< Cache search (does not require evals)
		SPEC_SEARCH      , ///< MADS speculative search (dynamic order in GPS)
		LH_SEARCH        , ///< Latin Hypercube (LH) search
		LH_SEARCH_P1     , ///< Latin Hypercube (LH) search during phase one
		MODEL_SEARCH     , ///< Model search
		VNS_SEARCH       , ///< VNS search
		P1_SEARCH        , ///< Phase one search
		ASYNCHRONOUS     , ///< Parallel asynchronous final evaluations
		USER_SEARCH      , ///< User search
		UNDEFINED_SEARCH   ///< Undefined search
    };
	
	/// Model type
	enum model_type
    {
		QUADRATIC_MODEL , ///< Quadratic model
        SGTELIB_MODEL   , ///< SGTELIB model
		NO_MODEL          ///< No models
    };
    
    /// intensification type
    enum intensification_type
    {
        NO_INTENSIFICATION ,
        POLL_ONLY , ///< Poll intensification only
        SEARCH_ONLY , ///< Search intensification only
        POLL_AND_SEARCH
    };
    
	/// Success type of an iteration
	// (do not modify this order)
	enum success_type
    {
		UNSUCCESSFUL    ,  ///< Failure
		PARTIAL_SUCCESS ,  ///< Partial success (improving)
        CACHE_UPDATE_SUCCESS, ///< Cache update success (RobustMads only) (dominating)
		FULL_SUCCESS,       ///< Full success (dominating)
        FULL_SUCCESS_ZOOM, ///< Full success (dominating)
        FULL_SUCCESS_STAY  ///< Full success (dominating)
    };
	
	/// Quadratic interpolation type
	enum interpolation_type
    {
		MFN                          , ///< Minimum Frobenius Norm interpolation.
		REGRESSION                   , ///< Regression.
		WP_REGRESSION                , ///< Well-poised regression.
		UNDEFINED_INTERPOLATION_TYPE   ///< Undefined.
    };
	
	/// Stopping criteria
	enum stop_type
    {
		NO_STOP                    ,  ///< No stop
		ERROR                      ,  ///< Error
		UNKNOWN_STOP_REASON        ,  ///< Unknown
		CTRL_C                     ,  ///< Ctrl-C
		USER_STOPPED               ,  ///< User-stopped in Evaluator::update_iteration()
		MESH_PREC_REACHED          ,  ///< Mesh minimum precision reached
		X0_FAIL                    ,  ///< Problem with starting point evaluation
		P1_FAIL                    ,  ///< Problem with phase one
		DELTA_M_MIN_REACHED        ,  ///< Min mesh size
		DELTA_P_MIN_REACHED        ,  ///< Min poll size
		L_MAX_REACHED              ,  ///< Max mesh index
        L_MIN_REACHED              ,  ///< Min mesh index
		L_LIMITS_REACHED           ,  ///< Mesh index limits
		XL_LIMITS_REACHED          ,  ///< Mesh index limits
   		GL_LIMITS_REACHED          ,  ///< Mesh index limits
		MAX_TIME_REACHED           ,  ///< Max time
		MAX_BB_EVAL_REACHED        ,  ///< Max number of blackbox evaluations
		MAX_BLOCK_EVAL_REACHED     ,  ///< Max number of block evaluations
		MAX_SGTE_EVAL_REACHED      ,  ///< Max number of surrogate evaluations
		MAX_EVAL_REACHED           ,  ///< Max number of evaluations
		MAX_SIM_BB_EVAL_REACHED    ,  ///< Max number of sim bb evaluations
		MAX_ITER_REACHED           ,  ///< Max number of iterations
		MAX_CONS_FAILED_ITER       ,  ///< Max number of consecutive failed iterations
		FEAS_REACHED               ,  ///< Feasibility
		F_TARGET_REACHED           ,  ///< F_TARGET
		STAT_SUM_TARGET_REACHED    ,  ///< STAT_SUM_TARGET
		L_CURVE_TARGET_REACHED     ,  ///< L_CURVE_TARGET
		MULTI_MAX_BB_REACHED       ,  ///< Max number of blackbox evaluations (multi obj.)
		MULTI_NB_MADS_RUNS_REACHED ,  ///< Max number of MADS runs (multi obj.)
		MULTI_STAGNATION           ,  ///< Stagnation criterion (multi obj.)
		MULTI_NO_PARETO_PTS        ,  ///< No Pareto points (multi obj.)
		MAX_CACHE_MEMORY_REACHED      ///< Max cache memory
    };
	
	/// Type of norm used for the computation of h
	enum hnorm_type
    {
		L1   ,  ///< norm L1
		L2   ,  ///< norm L2
		LINF    ///< norm Linf
    };
	
	/// Types of directions
	// (do not modify this order)
	enum direction_type
    {
		UNDEFINED_DIRECTION    , ///< Undefined direction
		MODEL_SEARCH_DIR       , ///< Model search direction
		NO_DIRECTION           , ///< No direction
		ORTHO_1                , ///< OrthoMADS 1
		ORTHO_2                , ///< OrthoMADS 2
		ORTHO_NP1_QUAD         , ///< OrthoMADS n+1 use Quad model to determine n+1-th direction 
		ORTHO_NP1_NEG          , ///< OrthoMADS n+1 use negative sum of n first directions to determine n+1-th direction
        ORTHO_NP1_UNI          , ///< OrthoMADS n+1 uniform directions
		DYN_ADDED              , ///< Dynamic addition (n+1-th direction added for ORTHO n+1)
		ORTHO_2N               , ///< OrthoMADS 2n
		LT_1                   , ///< LT-MADS 1
		LT_2                   , ///< LT-MADS 2
		LT_2N                  , ///< LT-MADS 2n
		LT_NP1                 , ///< LT-MADS n+1
		GPS_BINARY             , ///< GPS for binary variables
		GPS_2N_STATIC          , ///< GPS 2n static (classic coordinate search)
		GPS_2N_RAND            , ///< GPS 2n random
		GPS_NP1_STATIC_UNIFORM , ///< GPS n+1 static uniform
		GPS_NP1_STATIC         , ///< GPS n+1 static
		GPS_NP1_RAND_UNIFORM   , ///< GPS n+1 random uniform
		GPS_NP1_RAND           ,  ///< GPS n+1
		PROSPECT_DIR            ///< Prospect direction
    };
	
	/// Type for Eval_Point::check() failures
	enum check_failed_type
    {
		CHECK_OK     ,  ///< Correct check
		LB_FAIL      ,  ///< LB failure
		UB_FAIL      ,  ///< UB failure
		FIX_VAR_FAIL ,  ///< Fixed variables failure
		BIN_FAIL     ,  ///< Binary failure
		CAT_FAIL     ,  ///< Categorical failure
		INT_FAIL        ///< Integer failure
    };
	
	/// Type for cache indexes in Cache:
	enum cache_index_type
    {
		CACHE_1         ,  ///< Cache index #1
		CACHE_2         ,  ///< Cache index #2
		CACHE_3         ,  ///< Cache index #3
		UNDEFINED_CACHE    ///< Undefined cache index
    };
	
	/// Type for DISPLAY_STATS parameter
	// (do not modify this order):
	enum display_stats_type
    {
		DS_OBJ        ,    ///< Objective (f) value
		//   (keep in first position)
        DS_CONS_H     ,    ///< Infeasibility (h) value
        DS_SMOOTH_OBJ ,    ///< Smoothed objective value (f~)
		DS_SIM_BBE    ,    ///< Number of simulated bb evaluations
		DS_BBE        ,    ///< Number of bb evaluations
		DS_BLK_EVA	  ,    ///< Number of block evaluation calls	
		DS_SGTE       ,    ///< Number of surrogate evaluations
		DS_BBO        ,    ///< All blackbox outputs
		DS_EVAL       ,    ///< Number of evaluations
		DS_TIME       ,    ///< Wall-clock time
		DS_MESH_INDEX ,    ///< Mesh index
		DS_MESH_SIZE  ,    ///< Mesh size parameter Delta^m_k
		DS_DELTA_M    ,    ///< Same as \c DS_MESH_SIZE
		DS_POLL_SIZE  ,    ///< Poll size parameter Delta^p_k
		DS_DELTA_P    ,    ///< Same as \c DS_POLL_SIZE
		DS_SOL        ,    ///< Solution vector
		DS_VAR        ,    ///< One variable
		DS_STAT_SUM   ,    ///< Stat sum
		DS_STAT_AVG   ,    ///< Stat avg
		DS_UNDEFINED       ///< Und efined value
		//   (keep in last position)
    };
	
	/// Type for evaluation
	enum eval_type
    {
		TRUTH , ///< Truth
		SGTE    ///< Surrogate
    };
	
	/// Type for an evaluation status
	enum eval_status_type
    {
		EVAL_FAIL        ,  ///< Evaluation failure
		EVAL_USER_REJECT ,  ///< Evaluation was rejected by user (not failure -> may submitted again)
        EVAL_OK          ,  ///< Correct evaluation
		EVAL_IN_PROGRESS ,  ///< Evaluation in progress
		UNDEFINED_STATUS    ///< Undefined evaluation status
    };
    
    enum smoothing_status_type
    {
        SMOOTHING_FAIL   ,   ///< Smoothing failure
        SMOOTHING_OK     ,   ///< Smoothing success
        SMOOTHING_UNDEFINED  ///< Undefined smoothing status
    };
    
    // SGTELIB
    /// Formulations for sgtelib_model model Search
    enum sgtelib_model_formulation_type
    {
        SGTELIB_MODEL_FORMULATION_FS    ,  /// min f-lambda*sigma, st c-lambda*sigma < 0
        SGTELIB_MODEL_FORMULATION_FSP   ,  /// min f-lambda*sigma, st P(x) > 1/2
        SGTELIB_MODEL_FORMULATION_EIS   ,  /// min -EI-lambda*sigma, st c-lambda*sigma < 0
        SGTELIB_MODEL_FORMULATION_EFI   ,  /// min -EFI
        SGTELIB_MODEL_FORMULATION_EFIS  ,  /// min -EFI-lambda*sigma
        SGTELIB_MODEL_FORMULATION_EFIM  ,  /// min -EFI-lambda*sigma*mu
        SGTELIB_MODEL_FORMULATION_EFIC  ,  /// min -EFI-lambda*(EI*sigma+P*mu)
        SGTELIB_MODEL_FORMULATION_PFI   ,  /// min -PFI
        SGTELIB_MODEL_FORMULATION_D     ,  /// min -distance_to_closest
        SGTELIB_MODEL_FORMULATION_EXTERN,  /// min f, st c, with extern sgte.exe model
        SGTELIB_MODEL_FORMULATION_UNDEFINED /// Undefined
    };
    
    // SGTELIB
    /// Formulations for sgtelib_model model Search
    enum sgtelib_model_feasibility_type
    {
        SGTELIB_MODEL_FEASIBILITY_C         , /// one model for each constraint
        SGTELIB_MODEL_FEASIBILITY_H         , /// one model for H
        SGTELIB_MODEL_FEASIBILITY_B         , /// one binary model
        SGTELIB_MODEL_FEASIBILITY_M         , /// one model of the max of (c_j)
        SGTELIB_MODEL_FEASIBILITY_UNDEFINED   /// Undefined
    };
    
    enum mesh_type
    {
        XMESH , /// Anisotropic (eXtensible) mesh
        GMESH , /// Anisotropic Granular mesh
        SMESH ,  /// Isotropic Standard mesh
        NO_MESH_TYPE
    };
	
}

#endif
