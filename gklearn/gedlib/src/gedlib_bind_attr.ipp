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
 * @file gedlib_bind_attr.ipp
 * @brief implementations of classes and functions to call easily GebLib in Python without Gedlib's types
 */
#pragma once
//#ifndef SRC_GEDLIB_BIND_ATTR_IPP
//#define SRC_GEDLIB_BIND_ATTR_IPP

//Include standard libraries + GedLib library
// #include <iostream>
// #include "GedLibBind.h"
// #include "../include/gedlib-master/src/env/ged_env.hpp"
//#include "../include/gedlib-master/median/src/median_graph_estimator.hpp"
// #include "gedlib_bind_gxl.hpp"

using namespace std;

//Definition of types and templates used in this code for my human's memory :).
//ged::GEDEnv<UserNodeID, UserNodeLabel, UserEdgeLabel> env;
//template<class UserNodeID, class UserNodeLabel, class UserEdgeLabel> struct ExchangeGraph

//typedef std::map<std::string, std::string> GXLLabel;
//typedef std::string GXLNodeID;


namespace pyged {

//extern template class ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>;

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
	"GEOMETRIC",
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
	{"GEOMETRIC", ged::Options::EditCosts::GEOMETRIC},
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
//     // test only:
//     std::cout << "[gedlib_bind_attr.ipp] Available edit costs: ";
//     for (const auto& cost : editCostStringOptions) {
//         std::cout << cost << " ";
//     }

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


PyGEDEnvAttr::PyGEDEnvAttr () : env_(nullptr), initialized(false) {
    env_ = new ged::GEDEnv<ged::GXLNodeID, ged::AttrLabel, ged::AttrLabel>();
}

PyGEDEnvAttr::~PyGEDEnvAttr () {
	if (env_ != NULL) {
	    delete env_;
		env_ = NULL;
    }
}

// ======== Environment Public APIs ========

// bool initialized = false; //Initialization boolean (because Env has one but not accessible).

bool PyGEDEnvAttr::isInitialized() {
	return initialized;
}

void PyGEDEnvAttr::restartEnv() {
	if (env_ != NULL) {
		delete env_;
		env_ = NULL;
	}
	env_ = new ged::GEDEnv<ged::GXLNodeID, ged::AttrLabel, ged::AttrLabel>();
	initialized = false;
}

// void PyGEDEnvAttr::loadGXLGraph(const std::string & pathFolder, const std::string & pathXML, bool node_type, bool edge_type) {
// 	 std::vector<ged::GEDGraph::GraphID> tmp_graph_ids(env_->load_gxl_graph(pathFolder, pathXML,
// 	 	(node_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
// 		(edge_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
// 		std::unordered_set<std::string>(), std::unordered_set<std::string>()));
// }

std::pair<std::size_t,std::size_t> PyGEDEnvAttr::getGraphIds() const {
	return env_->graph_ids();
}

std::vector<std::size_t> PyGEDEnvAttr::getAllGraphIds() {
	std::vector<std::size_t> listID;
	for (std::size_t i = env_->graph_ids().first; i != env_->graph_ids().second; i++) {
		listID.push_back(i);
    }
	return listID;
}

const std::string PyGEDEnvAttr::getGraphClass(std::size_t id) const {
	return env_->get_graph_class(id);
}

const std::string PyGEDEnvAttr::getGraphName(std::size_t id) const {
	return env_->get_graph_name(id);
}

std::size_t PyGEDEnvAttr::addGraph(const std::string & graph_name, const std::string & graph_class) {
	ged::GEDGraph::GraphID newId = env_->add_graph(graph_name, graph_class);
	initialized = false;
	return std::stoi(std::to_string(newId));
}

// void PyGEDEnvAttr::addNode(std::size_t graphId, const std::string & nodeId, const std::map<std::string, std::string> & nodeLabel) {
// 	// todo: if this needs to be supported, we need to convert the string map to an AttrLabel
// 	env_->add_node(graphId, nodeId, nodeLabel);
// 	initialized = false;
// }

void PyGEDEnvAttr::addNode(
    std::size_t graphId,
    const std::string& nodeId,
    const std::unordered_map<std::string, std::string>& str_map,
    const std::unordered_map<std::string, int>& int_map,
    const std::unordered_map<std::string, double>& float_map,
    const std::unordered_map<std::string, std::vector<std::string>>& list_str_map,
    const std::unordered_map<std::string, std::vector<int>>& list_int_map,
    const std::unordered_map<std::string, std::vector<double>>& list_float_map
) {
//     // debug test only:
//     std::cout << "The node labels received by the c++ bindings are: " << std::endl;
//     printLabelMaps(str_map, int_map, float_map, list_str_map, list_int_map, list_float_map);

    // Merge the maps into AttrLabel:
    ged::AttrLabel nodeLabel = PyGEDEnvAttr::constructAttrLabelFromMaps(
        str_map,
        int_map,
        float_map,
        list_str_map,
        list_int_map,
        list_float_map
    );

//     std::cout << "The node label passed to c++ env is: " << std::endl;
//     printAttrLabel(nodeLabel);

    env_->add_node(graphId, nodeId, nodeLabel);
    initialized = false;
}

/*void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel) {
	env_->add_edge(graphId, tail, head, edgeLabel);
}*/

// void PyGEDEnvAttr::addEdge(std::size_t graphId, const std::string & tail, const std::string & head, const std::map<std::string, std::string> & edgeLabel, bool ignoreDuplicates) {
// 	// todo: if this needs to be supported, we need to convert the string map to an AttrLabel
// 	env_->add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
// 	initialized = false;
// }

void PyGEDEnvAttr::addEdge(
    std::size_t graphId,
    const std::string& tail,
    const std::string& head,
    const std::unordered_map<std::string, std::string>& str_map,
    const std::unordered_map<std::string, int>& int_map,
    const std::unordered_map<std::string, double>& float_map,
    const std::unordered_map<std::string, std::vector<std::string>>& list_str_map,
    const std::unordered_map<std::string, std::vector<int>>& list_int_map,
    const std::unordered_map<std::string, std::vector<double>>& list_float_map,
    bool ignoreDuplicates
) {
//     // debug test only:
//     std::cout << "The edge labels received by the c++ bindings are: " << std::endl;
//     printLabelMaps(str_map, int_map, float_map, list_str_map, list_int_map, list_float_map);

    // Merge the maps into AttrLabel:
    ged::AttrLabel edgeLabel = PyGEDEnvAttr::constructAttrLabelFromMaps(
        str_map,
        int_map,
        float_map,
        list_str_map,
        list_int_map,
        list_float_map
    );

//     std::cout << "The edge label passed to c++ env is: " << std::endl;
//     printAttrLabel(edgeLabel);

    env_->add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
    initialized = false;
}

void PyGEDEnvAttr::clearGraph(std::size_t graphId) {
	env_->clear_graph(graphId);
	initialized = false;
}

// todo: check if ExchangeGraph supports AttrLabel
ged::ExchangeGraph<ged::GXLNodeID, ged::AttrLabel, ged::AttrLabel> PyGEDEnvAttr::getGraph(std::size_t graphId) const {
//     static_assert(std::is_same_v<
//         decltype(env_->get_graph(graphId)),
//         ged::ExchangeGraph<ged::GXLNodeID, ged::AttrLabel, ged::AttrLabel>
//     >, "get_graph() 返回的不是 AttrLabel 类型");
//     std::cout << "get_graph() 返回的是 AttrLabel 类型: " << std::endl;


	return env_->get_graph(graphId);
}

std::size_t PyGEDEnvAttr::getGraphInternalId(std::size_t graphId) {
	return getGraph(graphId).id;
}

std::size_t PyGEDEnvAttr::getGraphNumNodes(std::size_t graphId) {
	return getGraph(graphId).num_nodes;
}

std::size_t PyGEDEnvAttr::getGraphNumEdges(std::size_t graphId) {
	return getGraph(graphId).num_edges;
}

std::vector<std::string> PyGEDEnvAttr::getGraphOriginalNodeIds(std::size_t graphId) {
	return getGraph(graphId).original_node_ids;
}

std::vector<ged::AttrLabel> PyGEDEnvAttr::getGraphNodeLabels(std::size_t graphId) {
	return getGraph(graphId).node_labels;
}

std::map<std::pair<std::size_t, std::size_t>, ged::AttrLabel> PyGEDEnvAttr::getGraphEdges(std::size_t graphId) {
	return getGraph(graphId).edge_labels;
}

std::vector<std::vector<std::size_t>> PyGEDEnvAttr::getGraphAdjacenceMatrix(std::size_t graphId) {
	return getGraph(graphId).adj_matrix;
}

void PyGEDEnvAttr::setEditCost(
    std::string editCost, std::vector<double> editCostConstants,
    std::unordered_map<std::string, std::string> editCostConfigStr,
    std::unordered_map<std::string, bool> editCostConfigBool
) {
    // Convert edit cost config values to std::any:
    std::unordered_map<std::string, std::any> editCostConstantsAny;
    for (const auto& pair : editCostConfigStr) {
        editCostConstantsAny[pair.first] = pair.second;
    }
    for (const auto& pair : editCostConfigBool) {
        editCostConstantsAny[pair.first] = pair.second;
    }

//     // debug test only:
//     std::cout << "[gedlib_bind_attr.ipp] edit cost config passed to C++ GEDEnv is: " << std::endl;
//     for (const auto& pair : editCostConstantsAny) {
//         std::cout << pair.first << ": ";
//         if (pair.second.type() == typeid(std::string)) {
//             std::cout << std::any_cast<std::string>(pair.second);
//         } else if (pair.second.type() == typeid(bool)) {
//             std::cout << std::any_cast<bool>(pair.second);
//         } else {
//             std::cout << "Unknown type";
//         }
//         std::cout << std::endl;
//     }

	env_->set_edit_costs(translateEditCost(editCost), editCostConstants, editCostConstantsAny);
}

void PyGEDEnvAttr::setPersonalEditCost(std::vector<double> editCostConstants) {
	//env_->set_edit_costs(Your EditCost Class(editCostConstants));
}

// void PyGEDEnvAttr::initEnv() {
// 	env_->init();
// 	initialized = true;
// }

void PyGEDEnvAttr::initEnv(std::string initOption, bool print_to_stdout) {
	env_->init(translateInitOptions(initOption), print_to_stdout);
	initialized = true;
}

void PyGEDEnvAttr::setMethod(std::string method, const std::string & options) {
	env_->set_method(translateMethod(method), options);
}

void PyGEDEnvAttr::initMethod() {
	env_->init_method();
}

double PyGEDEnvAttr::getInitime() const {
	return env_->get_init_time();
}

void PyGEDEnvAttr::runMethod(std::size_t g, std::size_t h) {
	env_->run_method(g, h);
}

double PyGEDEnvAttr::getUpperBound(std::size_t g, std::size_t h) const {
	return env_->get_upper_bound(g, h);
}

double PyGEDEnvAttr::getLowerBound(std::size_t g, std::size_t h) const {
	return env_->get_lower_bound(g, h);
}

std::vector<long unsigned int> PyGEDEnvAttr::getForwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_forward_map();
}

std::vector<long unsigned int> PyGEDEnvAttr::getBackwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_backward_map();
}

std::size_t PyGEDEnvAttr::getNodeImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).image(nodeId);
}

std::size_t PyGEDEnvAttr::getNodePreImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).pre_image(nodeId);
}

double PyGEDEnvAttr::getInducedCost(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).induced_cost();
}

std::vector<pair<std::size_t, std::size_t>> PyGEDEnvAttr::getNodeMap(std::size_t g, std::size_t h) {
	std::vector<pair<std::size_t, std::size_t>> res;
	std::vector<ged::NodeMap::Assignment> relation;
	env_->get_node_map(g, h).as_relation(relation);
	for (const auto & assignment : relation) {
		res.push_back(std::make_pair(assignment.first, assignment.second));
	}
	return res;
}

std::vector<std::vector<int>> PyGEDEnvAttr::getAssignmentMatrix(std::size_t g, std::size_t h) {
	std::vector<std::vector<int>> res;
	for(std::size_t i = 0; i != getForwardMap(g, h).size(); i++) {
		std::vector<int> newLine;
		bool have1 = false;
		for(std::size_t j = 0; j != getBackwardMap(g, h).size(); j++) {
			if (getNodeImage(g, h, i) == j) {
				newLine.push_back(1);
				have1 = true;
			}
			else{
				newLine.push_back(0);
			}
		}
		if(have1) {
			newLine.push_back(0);
		}
		else{
			newLine.push_back(1);
		}
		res.push_back(newLine);
	}
	std::vector<int> lastLine;
	for (size_t k = 0; k != getBackwardMap(g,h).size(); k++) {
		if (getBackwardMap(g,h)[k] ==  ged::GEDGraph::dummy_node()) {
			lastLine.push_back(1);
		}
		else{
			lastLine.push_back(0);
		}
	}
	res.push_back(lastLine);
	return res;
}

std::vector<std::vector<unsigned long int>> PyGEDEnvAttr::getAllMap(std::size_t g, std::size_t h) {
	std::vector<std::vector<unsigned long int>> res;
	res.push_back(getForwardMap(g, h));
	res.push_back(getBackwardMap(g,h));
	return res;
}

double PyGEDEnvAttr::getRuntime(std::size_t g, std::size_t h) const {
	return env_->get_runtime(g, h);
}

bool PyGEDEnvAttr::quasimetricCosts() const {
	return env_->quasimetric_costs();
}

std::vector<std::vector<size_t>> PyGEDEnvAttr::hungarianLSAP(std::vector<std::vector<std::size_t>> matrixCost) {
	std::size_t nrows = matrixCost.size();
	std::size_t ncols = matrixCost[0].size();
	std::size_t *rho = new std::size_t[nrows], *varrho = new std::size_t[ncols];
	std::size_t *u = new std::size_t[nrows], *v = new std::size_t[ncols];
	std::size_t *C = new std::size_t[nrows*ncols];
	// std::size_t i = 0, j;
	for (std::size_t i = 0; i < nrows; i++) {
		for (std::size_t j = 0; j < ncols; j++) {
			C[j*nrows+i] = matrixCost[i][j];
		}
	}
	lsape::hungarianLSAP<std::size_t>(C,nrows,ncols,rho,u,v,varrho);
	std::vector<std::vector<size_t>> res;
	res.push_back(translatePointer(rho, nrows));
	res.push_back(translatePointer(varrho, ncols));
	res.push_back(translatePointer(u, nrows));
	res.push_back(translatePointer(v, ncols));
	return res;
}

std::vector<std::vector<double>> PyGEDEnvAttr::hungarianLSAPE(std::vector<std::vector<double>> matrixCost) {
	std::size_t nrows = matrixCost.size();
	std::size_t ncols = matrixCost[0].size();
	std::size_t *rho = new std::size_t[nrows-1], *varrho = new std::size_t[ncols-1];
	double *u = new double[nrows], *v = new double[ncols];
	double *C = new double[nrows*ncols];
	for (std::size_t i = 0; i < nrows; i++) {
		for (std::size_t j = 0; j < ncols; j++) {
			C[j*nrows+i] = matrixCost[i][j];
		}
	}
	lsape::hungarianLSAPE<double,std::size_t>(C,nrows,ncols,rho,varrho,u,v);
	std::vector<std::vector<double>> res;
	res.push_back(translateAndConvertPointer(rho, nrows-1));
	res.push_back(translateAndConvertPointer(varrho, ncols-1));
	res.push_back(translatePointer(u, nrows));
	res.push_back(translatePointer(v, ncols));
	return res;
}

std::size_t PyGEDEnvAttr::getNumGraphs() const {
    return env_->num_graphs();
}

std::size_t PyGEDEnvAttr::getNumNodeLabels() const {
	return env_->num_node_labels();
}

ged::AttrLabel PyGEDEnvAttr::getNodeLabel(std::size_t label_id) const {
	return env_->get_node_label(label_id);
}

std::size_t PyGEDEnvAttr::getNumEdgeLabels() const {
	return env_->num_edge_labels();
}

ged::AttrLabel PyGEDEnvAttr::getEdgeLabel(std::size_t label_id) const {
	return env_->get_edge_label(label_id);
}

// std::size_t PyGEDEnvAttr::getNumNodes(std::size_t graph_id) const {
// 	return env_->get_num_nodes(graph_id);
// }

double PyGEDEnvAttr::getAvgNumNodes() const {
	return env_->get_avg_num_nodes();
}

double PyGEDEnvAttr::getNodeRelCost(const ged::AttrLabel & node_label_1, const ged::AttrLabel & node_label_2) const {
	return env_->node_rel_cost(node_label_1, node_label_2);
}

double PyGEDEnvAttr::getNodeDelCost(const ged::AttrLabel & node_label) const {
	return env_->node_del_cost(node_label);
}

double PyGEDEnvAttr::getNodeInsCost(const ged::AttrLabel & node_label) const {
	return env_->node_ins_cost(node_label);
}

ged::AttrLabel PyGEDEnvAttr::getMedianNodeLabel(const std::vector<ged::AttrLabel> & node_labels) const {
	return env_->median_node_label(node_labels);
}

double PyGEDEnvAttr::getEdgeRelCost(const ged::AttrLabel & edge_label_1, const ged::AttrLabel & edge_label_2) const {
	return env_->edge_rel_cost(edge_label_1, edge_label_2);
}

double PyGEDEnvAttr::getEdgeDelCost(const ged::AttrLabel & edge_label) const {
	return env_->edge_del_cost(edge_label);
}

double PyGEDEnvAttr::getEdgeInsCost(const ged::AttrLabel & edge_label) const {
	return env_->edge_ins_cost(edge_label);
}

ged::AttrLabel PyGEDEnvAttr::getMedianEdgeLabel(const std::vector<ged::AttrLabel> & edge_labels) const {
	return env_->median_edge_label(edge_labels);
}

std::string PyGEDEnvAttr::getInitType() const {
	return initOptionsToString(env_->get_init_type());
}

double PyGEDEnvAttr::computeInducedCost(std::size_t g_id, std::size_t h_id, std::vector<pair<std::size_t, std::size_t>> relation) const {
	ged::NodeMap node_map = ged::NodeMap(env_->get_num_nodes(g_id), env_->get_num_nodes(h_id));
	for (const auto & assignment : relation) {
		node_map.add_assignment(assignment.first, assignment.second);
		// std::cout << assignment.first << assignment.second << endl;
	}
	const std::vector<ged::GEDGraph::NodeID> forward_map = node_map.get_forward_map();
	for (std::size_t i{0}; i < node_map.num_source_nodes(); i++) {
		if (forward_map.at(i) == ged::GEDGraph::undefined_node()) {
			node_map.add_assignment(i, ged::GEDGraph::dummy_node());
		}
	}
	const std::vector<ged::GEDGraph::NodeID> backward_map = node_map.get_backward_map();
	for (std::size_t i{0}; i < node_map.num_target_nodes(); i++) {
		if (backward_map.at(i) == ged::GEDGraph::undefined_node()) {
			node_map.add_assignment(ged::GEDGraph::dummy_node(), i);
		}
	}
	// for (auto & map : node_map.get_forward_map()) {
	// 	std::cout << map << ", ";
	// }
	// std::cout << endl;
	// for (auto & map : node_map.get_backward_map()) {
	// 	std::cout << map << ", ";
	// }
	env_->compute_induced_cost(g_id, h_id, node_map);
	return node_map.induced_cost();
}


ged::AttrLabel
PyGEDEnvAttr::constructAttrLabelFromMaps(
    const std::unordered_map<std::string, std::string>& str_map,
    const std::unordered_map<std::string, int>& int_map,
    const std::unordered_map<std::string, double>& float_map,
    const std::unordered_map<std::string, std::vector<std::string>>& list_str_map,
    const std::unordered_map<std::string, std::vector<int>>& list_int_map,
    const std::unordered_map<std::string, std::vector<double>>& list_float_map
) {
    // using ged::AttrLabel = std::unordered_map<std::string, std::variant<std::string, int, double, std::vector<std::string>, std::vector<int>, std::vector<double>>>;
    ged::AttrLabel attr_label;
    for (const auto& pair : str_map) {
        attr_label[pair.first] = pair.second;
    }
    for (const auto& pair : int_map) {
        attr_label[pair.first] = pair.second;
    }
    for (const auto& pair : float_map) {
        attr_label[pair.first] = pair.second;
    }
    for (const auto& pair : list_str_map) {
        attr_label[pair.first] = pair.second;
    }
    for (const auto& pair : list_int_map) {
        attr_label[pair.first] = pair.second;
    }
    for (const auto& pair : list_float_map) {
        attr_label[pair.first] = pair.second;
    }
    return attr_label;
}


void printLabelMaps(
    const std::unordered_map<std::string, std::string>& str_map,
    const std::unordered_map<std::string, int>& int_map,
    const std::unordered_map<std::string, double>& float_map,
    const std::unordered_map<std::string, std::vector<std::string>>& list_str_map,
    const std::unordered_map<std::string, std::vector<int>>& list_int_map,
    const std::unordered_map<std::string, std::vector<double>>& list_float_map
) {
    // Print the label maps for debugging purposes
    std::cout << "String map: ";
    for (const auto& pair : str_map) {
        std::cout << pair.first << ": " << pair.second << ", ";
    }
    std::cout << "\nInt map: ";
    for (const auto& pair : int_map) {
        std::cout << pair.first << ": " << pair.second << ", ";
    }
    std::cout << "\nFloat map: ";
    for (const auto& pair : float_map) {
        std::cout << pair.first << ": " << pair.second << ", ";
    }
    std::cout << "\nList of strings map: ";
    for (const auto& pair : list_str_map) {
        std::cout << pair.first << ": [";
        for (const auto& item : pair.second) {
            std::cout << item << ", ";
        }
        std::cout << "], ";
    }
    std::cout << "\nList of ints map: ";
    for (const auto& pair : list_int_map) {
        std::cout << pair.first << ": [";
        for (const auto& item : pair.second) {
            std::cout << item << ", ";
        }
        std::cout << "], ";
    }
    std::cout << "\nList of floats map: ";
    for (const auto& pair : list_float_map) {
        std::cout << pair.first << ": [";
        for (const auto& item : pair.second) {
            std::cout << item << ", ";
        }
        std::cout << "], ";
    }

    std::cout << std::endl;

}

void printAttrLabel(const ged::AttrLabel & attr_label) {
    std::cout << "AttrLabel: ";
    for (const auto& pair : attr_label) {
        std::cout << pair.first << ": ";
        if (std::holds_alternative<std::string>(pair.second)) {
            std::cout << std::get<std::string>(pair.second);
        } else if (std::holds_alternative<int>(pair.second)) {
            std::cout << std::get<int>(pair.second);
        } else if (std::holds_alternative<double>(pair.second)) {
            std::cout << std::get<double>(pair.second);
        } else if (std::holds_alternative<std::vector<std::string>>(pair.second)) {
            const auto& vec = std::get<std::vector<std::string>>(pair.second);
            std::cout << "[";
            for (const auto& item : vec) {
                std::cout << item << ", ";
            }
            std::cout << "]";
        } else if (std::holds_alternative<std::vector<int>>(pair.second)) {
            const auto& vec = std::get<std::vector<int>>(pair.second);
            std::cout << "[";
            for (const auto& item : vec) {
                std::cout << item << ", ";
            }
            std::cout << "]";
        } else if (std::holds_alternative<std::vector<double>>(pair.second)) {
            const auto& vec = std::get<std::vector<double>>(pair.second);
            std::cout << "[";
            for (const auto& item : vec) {
                std::cout << item << ", ";
            }
            std::cout << "]";
        }
        std::cout << ", ";
    }
    std::cout << std::endl;
}

} // namespace pyged

//#endif /* SRC_GEDLIB_BIND_ATTR_IPP */
