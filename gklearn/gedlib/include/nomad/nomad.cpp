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
 \file   nomad.cpp
 \brief  NOMAD main file
 \author Sebastien Le Digabel
 \date   2010-04-12
 */
#include "nomad.hpp"

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main ( int argc , char ** argv )
{
    
    // display:
    NOMAD::Display out ( std::cout );
    out.precision ( NOMAD::DISPLAY_PRECISION_STD );
    
    
    std::string error;
    {
        // NOMAD initializations:
        NOMAD::begin ( argc , argv );
        
        // usage:
        if ( argc < 2 )
        {
            NOMAD::display_usage ( argv[0],std::cerr );
            NOMAD::end();
            return EXIT_FAILURE;
        }
        
        // parameters file:
        std::string opt             = argv[1];
        NOMAD::toupper ( opt );
        
        // display usage if option '-u' has been specified:
        if ( opt == "-U" || opt == "-USAGE" || opt == "--USAGE" )
        {
            NOMAD::display_usage ( argv[0], out );
            NOMAD::end();
            return EXIT_SUCCESS;
        }
        
        
        // display version if option '-v' has been specified:
        if ( opt == "-V" || opt == "-VERSION" || opt == "--VERSION" )
        {
            NOMAD::display_version ( out );
            NOMAD::end();
            return EXIT_SUCCESS;
        }
        
        // display info if option '-i' has been specified:
        if ( opt == "-I" || opt == "-INFO" || opt == "--INFO" )
        {
            NOMAD::display_info  ( out );
            NOMAD::display_usage ( argv[0], out );
            NOMAD::end();
            return EXIT_SUCCESS;
        }
        
        // parameters creation:
        NOMAD::Parameters p ( out );
        
        // display help on parameters if option '-h' has been specified:
        if ( opt == "-H" || opt == "-HELP" || opt == "--HELP" )
        {
            p.help ( argc , argv );
            NOMAD::end();
            return EXIT_SUCCESS;
        }
        
        // display developer help on parameters if option '-d' has been specified:
        if ( opt == "-D" || opt == "-DEVELOPER" || opt == "--DEVELOPER" )
        {
            p.help ( argc , argv,true );
            NOMAD::end();
            return EXIT_SUCCESS;
        }
        
        if ( opt[0] == '-' )
        {
            // Unrecognized flag. Display usage and exit gracefully.
            error = std::string("ERROR: Unrecognized option ") + argv[1];
            std::cerr << std::endl << error << std::endl << std::endl;
            NOMAD::display_usage ( argv[0], out );
            NOMAD::end();
            return EXIT_FAILURE;
        }

        std::string param_file_name = argv[1];

        // Verify the argument is a file name 
        if ( !NOMAD::check_read_file(param_file_name) )
        {
            // Could not read input file. Display usage and exit gracefully.
            error = std::string("ERROR: Could not read file \"") + argv[1] + "\"";
            std::cerr << std::endl << error << std::endl << std::endl;
            NOMAD::display_usage ( argv[0], out );
            NOMAD::end();
            return EXIT_FAILURE;
        }
        
        // check the number of processess:
#ifdef USE_MPI
        if ( NOMAD::Slave::get_nb_processes() < 2 )
        {
            error = std::string("ERROR: Incorrect command to run with MPI.");
            std::cerr << std::endl << error << std::endl << std::endl;
            NOMAD::display_usage ( argv[0], std::cerr );
            NOMAD::end();
            return EXIT_FAILURE;
        }
#endif
        
        try {

            // read parameters file:
            p.read ( param_file_name );
            
            // parameters check:
            p.check();
            
            // display NOMAD info and Seed:
            if ( p.get_display_degree() > NOMAD::MINIMAL_DISPLAY)
                NOMAD::display_info ( out );
            
            // parameters display:
            if ( NOMAD::Slave::is_master() &&
                p.get_display_degree() == NOMAD::FULL_DISPLAY )
                out << std::endl
                << NOMAD::open_block ( "parameters" ) << std::endl
                << p
                << NOMAD::close_block();
            
            // algorithm creation and execution:
            NOMAD::Mads mads ( p , NULL );
            if ( p.get_nb_obj() == 1 )
                mads.run();
            else
                mads.multi_run();
            
            
#ifdef MODEL_STATS
            mads.display_model_stats ( out );
#endif
            
        }
        catch ( std::exception & e )
        {
            if ( NOMAD::Slave::is_master() )
            {
                error = std::string ( "ERROR: " ) + e.what();
                std::cerr << std::endl << error << std::endl << std::endl;
            }
        }
        
        
        NOMAD::Slave::stop_slaves ( out );
        NOMAD::end();
        
    }
    
#ifdef MEMORY_DEBUG
    NOMAD::display_cardinalities ( out );
#endif
    
    return ( error.empty() ) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-----------------------------------------------------*/
/*  display NOMAD most important structures in memory  */
/*-----------------------------------------------------*/
#ifdef MEMORY_DEBUG
void NOMAD::display_cardinalities ( const NOMAD::Display & out )
{
    
#ifdef USE_MPI
    if ( !NOMAD::Slave::is_master() )
        return;
#endif
    
    // compute the biggest int value for appropriate display width:
    int max = (NOMAD::Double::get_max_cardinality() > NOMAD::Point::get_max_cardinality())
    ? NOMAD::Double::get_max_cardinality() : NOMAD::Point::get_max_cardinality();
    if ( NOMAD::Direction::get_max_cardinality() > max )
        max = NOMAD::Direction::get_max_cardinality();
    if ( NOMAD::Set_Element<NOMAD::Eval_Point>::get_max_cardinality() > max )
        max = NOMAD::Set_Element<NOMAD::Eval_Point>::get_max_cardinality();
    if ( NOMAD::Set_Element<NOMAD::Signature>::get_max_cardinality() > max )
        max = NOMAD::Set_Element<NOMAD::Signature>::get_max_cardinality();
    if ( NOMAD::Cache_File_Point::get_max_cardinality() > max )
        max = NOMAD::Cache_File_Point::get_max_cardinality();
    
    // cardinalities display:
    // ----------------------
    out << std::endl
    << NOMAD::open_block ( "important objects in memory" );
    
    // NOMAD::Signature:
    out << "Signature              : ";
    out.display_int_w ( NOMAD::Signature::get_cardinality() , max );
    out << " (max=";
    out.display_int_w ( NOMAD::Signature::get_max_cardinality() , max );
    out << ")" << std::endl;
    
    // NOMAD::Double:
    out << "Double                 : ";
    out.display_int_w ( NOMAD::Double::get_cardinality() , max );
    out << " (max=";
    out.display_int_w ( NOMAD::Double::get_max_cardinality() , max );
    out << ")" << std::endl;
    
    // NOMAD::Point:
    out << "Point                  : ";
    out.display_int_w ( NOMAD::Point::get_cardinality() , max );
    out << " (max=";
    out.display_int_w ( NOMAD::Point::get_max_cardinality() , max );
    out << ")" << std::endl;
    
    // NOMAD::Direction:
    out << "Direction              : ";
    out.display_int_w ( NOMAD::Direction::get_cardinality() , max );
    out << " (max=";
    out.display_int_w ( NOMAD::Direction::get_max_cardinality() , max );
    out << ")" << std::endl;
    
    // Set_Element<Eval_Point>:
    out << "Set_Element<Eval_Point>: ";
    out.display_int_w (NOMAD::Set_Element<NOMAD::Eval_Point>::get_cardinality(), max);
    out << " (max=";
    out.display_int_w (NOMAD::Set_Element<NOMAD::Eval_Point>::get_max_cardinality(), max);
    out << ")" << std::endl;
    
    // Set_Element<NOMAD::Signature>:
    out << "Set_Element<Signature> : ";
    out.display_int_w (NOMAD::Set_Element<NOMAD::Signature>::get_cardinality(), max);
    out << " (max=";
    out.display_int_w (NOMAD::Set_Element<NOMAD::Signature>::get_max_cardinality(), max);
    out << ")" << std::endl;
    
    // NOMAD::Cache_File_Point:
    out << "Cache_File_Point       : ";
    out.display_int_w ( NOMAD::Cache_File_Point::get_cardinality() , max );
    out << " (max=";
    out.display_int_w ( NOMAD::Cache_File_Point::get_max_cardinality() , max );
    out << ")" << std::endl;
    
    out << NOMAD::close_block();
}
#endif

/*------------------------------------------*/
/*            display NOMAD version         */
/*------------------------------------------*/
void NOMAD::display_version ( const NOMAD::Display & out )
{
#ifdef USE_MPI
    if ( !NOMAD::Slave::is_master() )
        return;
#endif
    out << std::endl << "NOMAD - version "
    << NOMAD::VERSION << " - www.gerad.ca/nomad"
    << std::endl << std::endl;
}

/*------------------------------------------*/
/*          display NOMAD information       */
/*------------------------------------------*/
void NOMAD::display_info ( const NOMAD::Display & out )
{
#ifdef USE_MPI
    if ( !NOMAD::Slave::is_master() )
        return;
#endif
    out << std::endl << "NOMAD - version "
    << NOMAD::VERSION
    << NOMAD::open_block(" has been created by")
    << "Charles Audet        - Ecole Polytechnique de Montreal" << std::endl
    << "Sebastien Le Digabel - Ecole Polytechnique de Montreal" << std::endl
    << "Christophe Tribes    - Ecole Polytechnique de Montreal" << std::endl
    << NOMAD::close_block()
    << std::endl
    << "The copyright of NOMAD - version "
    << NOMAD::VERSION
    << NOMAD::open_block(" is owned by")
    << "Sebastien Le Digabel - Ecole Polytechnique de Montreal" << std::endl
    << "Christophe Tribes    - Ecole Polytechnique de Montreal" << std::endl
    << NOMAD::close_block()
    << std::endl
    << "NOMAD v3 has been funded by AFOSR, Exxon Mobil, Hydro Québec, Rio Tinto and " << std::endl
    << "IVADO." << std::endl
    << std::endl
    << "NOMAD v3 is a new version of NOMAD v1 and v2. NOMAD v1 and v2 were created" << std::endl
    << "and developed by Mark Abramson, Charles Audet, Gilles Couture, and John E." << std::endl
    << "Dennis Jr., and were funded by AFOSR and Exxon Mobil." << std::endl
    << std::endl
    << "License   : \'" << NOMAD::LGPL_FILE       << "\'" << std::endl
    << "User guide: \'" << NOMAD::USER_GUIDE_FILE << "\'" << std::endl
    << "Examples  : \'" << NOMAD::EXAMPLES_DIR    << "\'" << std::endl
    << "Tools     : \'" << NOMAD::TOOLS_DIR       << "\'" << std::endl
    << std::endl
    << "Please report bugs to nomad@gerad.ca"
    << std::endl;
    out << endl << "Seed: "<< NOMAD::RNG::get_seed()<<endl;}

/*------------------------------------------*/
/*             display NOMAD usage          */
/*------------------------------------------*/
void NOMAD::display_usage ( char* exeName, const NOMAD::Display & out )
{
#ifdef USE_MPI
    if ( !NOMAD::Slave::is_master() )
        return;
    out << std::endl
    << "Run NOMAD.MPI  : mpirun -np p " << exeName << " parameters_file" << std::endl
    << "Info           : " << exeName << " -i"                           << std::endl
    << "Help           : " << exeName << " -h keyword(s) (or 'all')"     << std::endl
    << "Developer help : " << exeName << " -d keyword(s) (or 'all')"     << std::endl
    << "Version        : " << exeName << " -v"                           << std::endl
    << "Usage          : " << exeName << " -u"                          << std::endl
    << std::endl;  
#else
    out << std::endl
    << "Run NOMAD      : " << exeName << " parameters_file"          << std::endl
    << "Info           : " << exeName << " -i"                       << std::endl
    << "Help           : " << exeName << " -h keyword(s) (or 'all')" << std::endl
    << "Developer help : " << exeName << " -d keyword(s) (or 'all')" << std::endl
    << "Version        : " << exeName << " -v"                       << std::endl
    << "Usage          : " << exeName << " -u"                       << std::endl
    << std::endl; 
#endif
}
