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
 * @file gedlib_bind_util.cpp
 * @brief implementation of util class and function declarations to call easily GedLib in Python
 */
#pragma once
//#ifndef GEDLIB_BIND_UTIL_CPP
//#define GEDLIB_BIND_UTIL_CPP

//Include standard libraries + GedLib library
// #include <iostream>
// #include "GedLibBind.h"
// #include "../include/gedlib-master/src/env/ged_env.hpp"
//#include "../include/gedlib-master/median/src/median_graph_estimator.hpp"
#include "gedlib_bind_util.hpp"

using namespace std;

//Definition of types and templates used in this code for my human's memory :).
//ged::GEDEnv<UserNodeID, UserNodeLabel, UserEdgeLabel> env;
//template<class UserNodeID, class UserNodeLabel, class UserEdgeLabel> struct ExchangeGraph

//typedef std::map<std::string, std::string> GXLLabel;
//typedef std::string GXLNodeID;


namespace pyged {

//!< List of available edit cost functions readable by Python.
std::vector<std::string> editCostStringOptions = {
	"CHEM_1",
	"CHEM_2",
	"CMU",
	"GREC_1",
	"GREC_2",
	"LETTER",
	"LETTER2",
	"NON_SYMBOLIC",
	"FINGERPRINT",
	"PROTEIN",
	"CONSTANT"
};

//!< Map of available edit cost functions between enum type in C++ and string in Python
std::map<std::string, ged::Options::EditCosts> editCostOptions = {
	{"CHEM_1", ged::Options::EditCosts::CHEM_1},
	{"CHEM_2", ged::Options::EditCosts::CHEM_2},
	{"CMU", ged::Options::EditCosts::CMU},
	{"GREC_1", ged::Options::EditCosts::GREC_1},
	{"GREC_2", ged::Options::EditCosts::GREC_2},
	{"LETTER", ged::Options::EditCosts::LETTER},
	{"LETTER2", ged::Options::EditCosts::LETTER2},
	{"NON_SYMBOLIC", ged::Options::EditCosts::NON_SYMBOLIC},
	{"FINGERPRINT", ged::Options::EditCosts::FINGERPRINT},
	{"PROTEIN", ged::Options::EditCosts::PROTEIN},
	{"CONSTANT", ged::Options::EditCosts::CONSTANT}
};

 //!< List of available computation methods readable by Python.
std::vector<std::string> methodStringOptions = {
	"BRANCH",
	"BRANCH_FAST",
	"BRANCH_TIGHT",
	"BRANCH_UNIFORM",
	"BRANCH_COMPACT",
	"PARTITION",
	"HYBRID",
	"RING",
	"ANCHOR_AWARE_GED",
	"WALKS",
	"IPFP",
	"BIPARTITE",
	"SUBGRAPH",
	"NODE",
	"RING_ML",
	"BIPARTITE_ML",
	"REFINE",
	"BP_BEAM",
	"SIMULATED_ANNEALING",
	"HED",
	"STAR"
};

//!< Map of available computation methods readables between enum type in C++ and string in Python
std::map<std::string, ged::Options::GEDMethod> methodOptions = {
	{"BRANCH", ged::Options::GEDMethod::BRANCH},
	{"BRANCH_FAST", ged::Options::GEDMethod::BRANCH_FAST},
	{"BRANCH_TIGHT", ged::Options::GEDMethod::BRANCH_TIGHT},
	{"BRANCH_UNIFORM", ged::Options::GEDMethod::BRANCH_UNIFORM},
	{"BRANCH_COMPACT", ged::Options::GEDMethod::BRANCH_COMPACT},
	{"PARTITION", ged::Options::GEDMethod::PARTITION},
	{"HYBRID", ged::Options::GEDMethod::HYBRID},
	{"RING", ged::Options::GEDMethod::RING},
	{"ANCHOR_AWARE_GED", ged::Options::GEDMethod::ANCHOR_AWARE_GED},
	{"WALKS", ged::Options::GEDMethod::WALKS},
	{"IPFP", ged::Options::GEDMethod::IPFP},
	{"BIPARTITE", ged::Options::GEDMethod::BIPARTITE},
	{"SUBGRAPH", ged::Options::GEDMethod::SUBGRAPH},
	{"NODE", ged::Options::GEDMethod::NODE},
	{"RING_ML", ged::Options::GEDMethod::RING_ML},
	{"BIPARTITE_ML",ged::Options::GEDMethod::BIPARTITE_ML},
	{"REFINE",ged::Options::GEDMethod::REFINE},
	{"BP_BEAM", ged::Options::GEDMethod::BP_BEAM},
	{"SIMULATED_ANNEALING", ged::Options::GEDMethod::SIMULATED_ANNEALING},
	{"HED", ged::Options::GEDMethod::HED},
	{"STAR"	, ged::Options::GEDMethod::STAR},
};

//!<List of available initilaization options readable by Python.
std::vector<std::string> initStringOptions = {
	"LAZY_WITHOUT_SHUFFLED_COPIES",
	"EAGER_WITHOUT_SHUFFLED_COPIES",
	"LAZY_WITH_SHUFFLED_COPIES",
	"EAGER_WITH_SHUFFLED_COPIES"
};

//!< Map of available initilaization options readables between enum type in C++ and string in Python
std::map<std::string, ged::Options::InitType> initOptions = {
	{"LAZY_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITHOUT_SHUFFLED_COPIES},
	{"EAGER_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITHOUT_SHUFFLED_COPIES},
	{"LAZY_WITH_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITH_SHUFFLED_COPIES},
	{"EAGER_WITH_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITH_SHUFFLED_COPIES}
};

std::vector<std::string> getEditCostStringOptions() {
	return editCostStringOptions;
}

std::vector<std::string> getMethodStringOptions() {
	return methodStringOptions;
}

std::vector<std::string> getInitStringOptions() {
	return initStringOptions;
}

static std::size_t getDummyNode() {
	return ged::GEDGraph::dummy_node();
}


/*!
 * @brief Returns the enum EditCost which correspond to the string parameter
 * @param editCost Select one of the predefined edit costs in the list.
 * @return The edit cost function which correspond in the edit cost functions map.
 */
ged::Options::EditCosts translateEditCost(std::string editCost) {
	 for (std::size_t i = 0; i != editCostStringOptions.size(); i++) {
		 if (editCostStringOptions[i] == editCost) {
			 return editCostOptions[editCostStringOptions[i]];
		 }
	 }
	 return ged::Options::EditCosts::CONSTANT;
}

/*!
 * @brief Returns the enum IniType which correspond to the string parameter
 * @param initOption Select initialization options.
 * @return The init Type which correspond in the init options map.
 */
ged::Options::InitType translateInitOptions(std::string initOption) {
	 for (std::size_t i = 0; i != initStringOptions.size(); i++) {
		 if (initStringOptions[i] == initOption) {
			 return initOptions[initStringOptions[i]];
		 }
	 }
	 return ged::Options::InitType::EAGER_WITHOUT_SHUFFLED_COPIES;
}

/*!
 * @brief Returns the string correspond to the enum IniType.
 * @param initOption Select initialization options.
 * @return The string which correspond to the enum IniType @p initOption.
 */
 std::string initOptionsToString(ged::Options::InitType initOption) {
	 for (std::size_t i = 0; i != initOptions.size(); i++) {
		 if (initOptions[initStringOptions[i]] == initOption) {
			 return initStringOptions[i];
		 }
	 }
	 return "EAGER_WITHOUT_SHUFFLED_COPIES";
}

/*!
 * @brief Returns the enum Method which correspond to the string parameter
 * @param method Select the method that is to be used.
 * @return The computation method which correspond in the edit cost functions map.
 */
ged::Options::GEDMethod translateMethod(std::string method) {
	 for (std::size_t i = 0; i != methodStringOptions.size(); i++) {
		 if (methodStringOptions[i] == method) {
			 return methodOptions[methodStringOptions[i]];
		 }
	 }
	 return ged::Options::GEDMethod::STAR;
}

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The size_t pointer to convert.
 * @return The vector which contains the pointer's values.
 */
std::vector<size_t> translatePointer(std::size_t* pointer, std::size_t dataSize ) {
	std::vector<size_t> res;
	for(std::size_t i = 0; i < dataSize; i++) {
		res.push_back(pointer[i]);
	}
	return res;
}

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The double pointer to convert.
 * @return The vector which contains the pointer's values.
 */
std::vector<double> translatePointer(double* pointer, std::size_t dataSize ) {
	std::vector<double> res;
	for(std::size_t i = 0; i < dataSize; i++) {
		res.push_back(pointer[i]);
	}
	return res;
}

/*!
 * @brief Returns the vector of values which correspond to the pointer parameter.
 * @param pointer The size_t pointer to convert.
 * @return The vector which contains the pointer's values, with double type.
 */
std::vector<double> translateAndConvertPointer(std::size_t* pointer, std::size_t dataSize ) {
	std::vector<double> res;
	for(std::size_t i = 0; i < dataSize; i++) {
		res.push_back((double)pointer[i]);
	}
	return res;
}

/*!
 * @brief Returns the string which contains all element of a int list.
 * @param vector The vector to translate.
 * @return The string which contains all elements separated with a blank space.
 */
std::string toStringVectorInt(std::vector<int> vector) {
	std::string res = "";

    for (std::size_t i = 0; i != vector.size(); i++)
    {
       res += std::to_string(vector[i]) + " ";
    }

    return res;
}

/*!
 * @brief Returns the string which contains all element of a unsigned long int list.
 * @param vector The vector to translate.
 * @return The string which contains all elements separated with a blank space.
 */
std::string toStringVectorInt(std::vector<unsigned long int> vector) {
	std::string res = "";

    for (std::size_t i = 0; i != vector.size(); i++)
    {
        res += std::to_string(vector[i]) + " ";
    }

    return res;
}

/*void medianLetter(pathFolder, pathXML, editCost, method, options="", initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") {

	if(isInitialized()) {
		restartEnv();
	}
	setEditCost(editCost);*/

	/*std::string letter_class("A");
	if (argc > 1) {
		letter_class = std::string(argv[1]);
	}*/
	//std::string seed("0");
	/*if (argc > 2) {
		seed = std::string(argv[2]);
	}*/

	/*loadGXLGraph(pathFolder, pathXML);
	std::vector<std::size_t> graph_ids = getAllGraphIds();
	std::size_t median_id = env_->add_graph("median", "");

	initEnv(initOption);

	setMethod(method);

	ged::MedianGraphEstimator<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> median_estimator(&env, false);
	median_estimator.set_options("--init-type RANDOM --randomness PSEUDO --seed " + seed);
	median_estimator.run(graph_ids, median_id);
	std::string gxl_file_name("../output/gen_median_Letter_HIGH_" + letter_class + ".gxl");
	env_->save_as_gxl_graph(median_id, gxl_file_name);*/

	/*std::string tikz_file_name("../output/gen_median_Letter_HIGH_" + letter_class + ".tex");
	save_letter_graph_as_tikz_file(env_->get_graph(median_id), tikz_file_name);*/
//}

} // namespace pyged

//#endif /* GEDLIB_BIND_UTIL_CPP */



// namespace shapes {

//     // Default constructor
//     Rectangle::Rectangle () {}

//     // Overloaded constructor
//     Rectangle::Rectangle (int x0, int y0, int x1, int y1) {
//         this->x0 = x0;
//         this->y0 = y0;
//         this->x1 = x1;
//         this->y1 = y1;
//     }

//     // Destructor
//     Rectangle::~Rectangle () {}

//     // Return the area of the rectangle
//     int Rectangle::getArea () {
//         return (this->x1 - this->x0) * (this->y1 - this->y0);
//     }

//     // Get the size of the rectangle.
//     // Put the size in the pointer args
//     void Rectangle::getSize (int *width, int *height) {
//         (*width) = x1 - x0;
//         (*height) = y1 - y0;
//     }

//     // Move the rectangle by dx dy
//     void Rectangle::move (int dx, int dy) {
//         this->x0 += dx;
//         this->y0 += dy;
//         this->x1 += dx;
//         this->y1 += dy;
//     }
// }