/****************************************************************************
 *                                                                          *
 *   Copyright (C) 2019-2025 by Linlin Jia, Natacha Lambert, and David B.   *
 *   Blumenthal                                                             *
 *                                                                          *
 *   This file should be used by Python.                                    *
 * 	 Please call the Python module if you want to use GedLib with this code.*
 *                                                                          *
 * 	 Otherwise, you can directly use GedLib for C++.                        *
 *                                                                          *
 ***************************************************************************/

/*!
 * @file gedlib_bind_util.hpp
 * @brief Util class and function declarations to call easily GebLib in Python
 */
#pragma once
//#ifndef GEDLIB_BIND_UTIL_HPP
//#define GEDLIB_BIND_UTIL_HPP

//Include standard libraries.
#include <string>
#include <vector>
#include <iostream>
//#include "../include/gedlib-master/src/env/ged_env.hpp"
//#include "../include/gedlib-master/src/env/node_map.hpp"
#include "gedlib_header.hpp"

/*!
 * @namespace pyged
 * @brief Global namespace for gedlibpy.
 */
namespace pyged {
//
////!< List of available edit cost functions readable by Python.
//extern std::vector<std::string> editCostStringOptions;
//
////!< Map of available edit cost functions between enum type in C++ and string in Python
//extern std::map<std::string, ged::Options::EditCosts> editCostOptions;
//
// //!< List of available computation methods readable by Python.
//extern std::vector<std::string> methodStringOptions;
//
////!< Map of available computation methods readables between enum type in C++ and string in Python
//extern std::map<std::string, ged::Options::GEDMethod> methodOptions;
//
////!<List of available initilaization options readable by Python.
//extern std::vector<std::string> initStringOptions;
//
////!< Map of available initilaization options readables between enum type in C++ and string in Python
//extern std::map<std::string, ged::Options::InitType> initOptions;


/*!
* @brief Get list of available edit cost functions readable by Python.
*/
std::vector<std::string> getEditCostStringOptions();

/*!
* @brief Get list of available computation methods readable by Python.
*/
std::vector<std::string> getMethodStringOptions();

/*!
* @brief Get list of available initilaization options readable by Python.
*/
std::vector<std::string> getInitStringOptions();

/*!
* @brief Returns a dummy node.
* @return ID of dummy node.
*/
static std::size_t getDummyNode();

/*!
 * @brief Returns the enum EditCost which correspond to the string parameter
 * @param editCost Select one of the predefined edit costs in the list.
 * @return The edit cost function which correspond in the edit cost functions map.
 */
ged::Options::EditCosts translateEditCost(std::string editCost);


/*!
 * @brief Returns the enum IniType which correspond to the string parameter
 * @param initOption Select initialization options.
 * @return The init Type which correspond in the init options map.
 */
ged::Options::InitType translateInitOptions(std::string initOption);

/*!
 * @brief Returns the string correspond to the enum IniType.
 * @param initOption Select initialization options.
 * @return The string which correspond to the enum IniType @p initOption.
 */
 std::string initOptionsToString(ged::Options::InitType initOption);

 /*!
 * @brief Returns the enum Method which correspond to the string parameter
 * @param method Select the method that is to be used.
 * @return The computation method which correspond in the edit cost functions map.
 */
ged::Options::GEDMethod translateMethod(std::string method);

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The size_t pointer to convert.
 * @return The vector which contains the pointer's values.
 */
std::vector<size_t> translatePointer(std::size_t* pointer, std::size_t dataSize );

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The double pointer to convert.
 * @return The vector which contains the pointer's values.
 */
std::vector<double> translatePointer(double* pointer, std::size_t dataSize );

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The size_t pointer to convert.
 * @return The vector which contains the pointer's values, with double type.
 */
std::vector<double> translateAndConvertPointer(std::size_t* pointer, std::size_t dataSize );

/*!
 * @brief Returns the string which contains all element of a int list.
 * @param vector The vector to translate.
 * @return The string which contains all elements separated with a blank space.
 */
std::string toStringVectorInt(std::vector<int> vector);

/*!
 * @brief Returns the string which contains all element of a unsigned long int list.
 * @param vector The vector to translate.
 * @return The string which contains all elements separated with a blank space.
 */
std::string toStringVectorInt(std::vector<unsigned long int> vector);

} // namespace pyged

//#endif /* GEDLIB_BIND_UTIL_HPP */