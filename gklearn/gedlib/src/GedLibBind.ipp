/****************************************************************************
 *                                                                          *
 *   Copyright (C) 2019-2020 by Natacha Lambert, David B. Blumenthal and    *
 *   Linlin Jia                                                             *
 *                                                                          *
 *   This file should be used by Python.                                    *
 * 	 Please call the Python module if you want to use GedLib with this code.*
 *                                                                          *
 * 	 Otherwise, you can directly use GedLib for C++.                        *
 *                                                                          *
 ***************************************************************************/

/*!
 * @file GedLibBind.ipp
 * @brief Classe and function definitions to call easly GebLib in Python without Gedlib's types
 */
#ifndef GEDLIBBIND_IPP
#define GEDLIBBIND_IPP

//Include standard libraries + GedLib library
// #include <iostream>
// #include "GedLibBind.h"
// #include "../include/gedlib-master/src/env/ged_env.hpp"
//#include "../include/gedlib-master/median/src/median_graph_estimator.hpp"

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


PyGEDEnv::PyGEDEnv () {
	env_ = new ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	this->initialized = false;
}

PyGEDEnv::~PyGEDEnv () {
	if (env_ != NULL) {
		delete env_;
		env_ = NULL;
	}
}

// bool initialized = false; //Initialization boolean (because Env has one but not accessible).

bool PyGEDEnv::isInitialized() {
	return initialized;
}

void PyGEDEnv::restartEnv() {
	if (env_ != NULL) {
		delete env_;
		env_ = NULL;
	}
	env_ = new ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	initialized = false;
}

void PyGEDEnv::loadGXLGraph(const std::string & pathFolder, const std::string & pathXML, bool node_type, bool edge_type) {
	 std::vector<ged::GEDGraph::GraphID> tmp_graph_ids(env_->load_gxl_graph(pathFolder, pathXML, 
	 	(node_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
		(edge_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
		std::unordered_set<std::string>(), std::unordered_set<std::string>()));
}

std::pair<std::size_t,std::size_t> PyGEDEnv::getGraphIds() const {
	return env_->graph_ids();
}

std::vector<std::size_t> PyGEDEnv::getAllGraphIds() {
	std::vector<std::size_t> listID;
	for (std::size_t i = env_->graph_ids().first; i != env_->graph_ids().second; i++) {
		listID.push_back(i);
    }
	return listID;
}

const std::string PyGEDEnv::getGraphClass(std::size_t id) const {
	return env_->get_graph_class(id);
}

const std::string PyGEDEnv::getGraphName(std::size_t id) const {
	return env_->get_graph_name(id);
}

std::size_t PyGEDEnv::addGraph(const std::string & graph_name, const std::string & graph_class) {
	ged::GEDGraph::GraphID newId = env_->add_graph(graph_name, graph_class);
	initialized = false;
	return std::stoi(std::to_string(newId));
}

void PyGEDEnv::addNode(std::size_t graphId, const std::string & nodeId, const std::map<std::string, std::string> & nodeLabel) {
	env_->add_node(graphId, nodeId, nodeLabel);
	initialized = false;
}

/*void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel) {
	env_->add_edge(graphId, tail, head, edgeLabel);
}*/

void PyGEDEnv::addEdge(std::size_t graphId, const std::string & tail, const std::string & head, const std::map<std::string, std::string> & edgeLabel, bool ignoreDuplicates) {
	env_->add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
	initialized = false;
}

void PyGEDEnv::clearGraph(std::size_t graphId) {
	env_->clear_graph(graphId);
	initialized = false;
}

ged::ExchangeGraph<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> PyGEDEnv::getGraph(std::size_t graphId) const {
	return env_->get_graph(graphId);
}

std::size_t PyGEDEnv::getGraphInternalId(std::size_t graphId) {
	return getGraph(graphId).id;
}

std::size_t PyGEDEnv::getGraphNumNodes(std::size_t graphId) {
	return getGraph(graphId).num_nodes;
}

std::size_t PyGEDEnv::getGraphNumEdges(std::size_t graphId) {
	return getGraph(graphId).num_edges;
}

std::vector<std::string> PyGEDEnv::getGraphOriginalNodeIds(std::size_t graphId) {
	return getGraph(graphId).original_node_ids;
}

std::vector<std::map<std::string, std::string>> PyGEDEnv::getGraphNodeLabels(std::size_t graphId) {
	return getGraph(graphId).node_labels;
}

std::map<std::pair<std::size_t, std::size_t>, std::map<std::string, std::string>> PyGEDEnv::getGraphEdges(std::size_t graphId) {
	return getGraph(graphId).edge_labels;
}

std::vector<std::vector<std::size_t>> PyGEDEnv::getGraphAdjacenceMatrix(std::size_t graphId) {
	return getGraph(graphId).adj_matrix;
}

void PyGEDEnv::setEditCost(std::string editCost, std::vector<double> editCostConstants) {
	env_->set_edit_costs(translateEditCost(editCost), editCostConstants);
}

void PyGEDEnv::setPersonalEditCost(std::vector<double> editCostConstants) {
	//env_->set_edit_costs(Your EditCost Class(editCostConstants));
}

// void PyGEDEnv::initEnv() {
// 	env_->init();
// 	initialized = true;
// }

void PyGEDEnv::initEnv(std::string initOption, bool print_to_stdout) {
	env_->init(translateInitOptions(initOption), print_to_stdout);
	initialized = true;
}

void PyGEDEnv::setMethod(std::string method, const std::string & options) {
	env_->set_method(translateMethod(method), options);
}

void PyGEDEnv::initMethod() {
	env_->init_method();
}

double PyGEDEnv::getInitime() const {
	return env_->get_init_time();
}

void PyGEDEnv::runMethod(std::size_t g, std::size_t h) {
	env_->run_method(g, h);
}

double PyGEDEnv::getUpperBound(std::size_t g, std::size_t h) const {
	return env_->get_upper_bound(g, h);
}

double PyGEDEnv::getLowerBound(std::size_t g, std::size_t h) const {
	return env_->get_lower_bound(g, h);
}

std::vector<long unsigned int> PyGEDEnv::getForwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_forward_map();
}

std::vector<long unsigned int> PyGEDEnv::getBackwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_backward_map();
}

std::size_t PyGEDEnv::getNodeImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).image(nodeId);
}

std::size_t PyGEDEnv::getNodePreImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).pre_image(nodeId);
}

double PyGEDEnv::getInducedCost(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).induced_cost();
}

std::vector<pair<std::size_t, std::size_t>> PyGEDEnv::getNodeMap(std::size_t g, std::size_t h) {
	std::vector<pair<std::size_t, std::size_t>> res;
	std::vector<ged::NodeMap::Assignment> relation;
	env_->get_node_map(g, h).as_relation(relation);
	for (const auto & assignment : relation) {
		res.push_back(std::make_pair(assignment.first, assignment.second));
	}
	return res;
}

std::vector<std::vector<int>> PyGEDEnv::getAssignmentMatrix(std::size_t g, std::size_t h) {
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

std::vector<std::vector<unsigned long int>> PyGEDEnv::getAllMap(std::size_t g, std::size_t h) {
	std::vector<std::vector<unsigned long int>> res;
	res.push_back(getForwardMap(g, h));
	res.push_back(getBackwardMap(g,h));
	return res;
}

double PyGEDEnv::getRuntime(std::size_t g, std::size_t h) const {
	return env_->get_runtime(g, h);
}

bool PyGEDEnv::quasimetricCosts() const {
	return env_->quasimetric_costs();
}

std::vector<std::vector<size_t>> PyGEDEnv::hungarianLSAP(std::vector<std::vector<std::size_t>> matrixCost) {
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

std::vector<std::vector<double>> PyGEDEnv::hungarianLSAPE(std::vector<std::vector<double>> matrixCost) {
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

std::size_t PyGEDEnv::getNumNodeLabels() const {
	return env_->num_node_labels();
}

std::map<std::string, std::string> PyGEDEnv::getNodeLabel(std::size_t label_id) const {
	return env_->get_node_label(label_id);
}

std::size_t PyGEDEnv::getNumEdgeLabels() const {
	return env_->num_edge_labels();
}

std::map<std::string, std::string> PyGEDEnv::getEdgeLabel(std::size_t label_id) const {
	return env_->get_edge_label(label_id);
}

// std::size_t PyGEDEnv::getNumNodes(std::size_t graph_id) const {
// 	return env_->get_num_nodes(graph_id);
// }

double PyGEDEnv::getAvgNumNodes() const {
	return env_->get_avg_num_nodes();
}

double PyGEDEnv::getNodeRelCost(const std::map<std::string, std::string> & node_label_1, const std::map<std::string, std::string> & node_label_2) const {
	return env_->node_rel_cost(node_label_1, node_label_2);
}

double PyGEDEnv::getNodeDelCost(const std::map<std::string, std::string> & node_label) const {
	return env_->node_del_cost(node_label);
}

double PyGEDEnv::getNodeInsCost(const std::map<std::string, std::string> & node_label) const {
	return env_->node_ins_cost(node_label);
}

std::map<std::string, std::string> PyGEDEnv::getMedianNodeLabel(const std::vector<std::map<std::string, std::string>> & node_labels) const {
	return env_->median_node_label(node_labels);
}

double PyGEDEnv::getEdgeRelCost(const std::map<std::string, std::string> & edge_label_1, const std::map<std::string, std::string> & edge_label_2) const {
	return env_->edge_rel_cost(edge_label_1, edge_label_2);
}

double PyGEDEnv::getEdgeDelCost(const std::map<std::string, std::string> & edge_label) const {
	return env_->edge_del_cost(edge_label);
}

double PyGEDEnv::getEdgeInsCost(const std::map<std::string, std::string> & edge_label) const {
	return env_->edge_ins_cost(edge_label);
}

std::map<std::string, std::string> PyGEDEnv::getMedianEdgeLabel(const std::vector<std::map<std::string, std::string>> & edge_labels) const {
	return env_->median_edge_label(edge_labels);
}

std::string PyGEDEnv::getInitType() const {
	return initOptionsToString(env_->get_init_type());
}

double PyGEDEnv::computeInducedCost(std::size_t g_id, std::size_t h_id, std::vector<pair<std::size_t, std::size_t>> relation) const {
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




// double PyGEDEnv::getNodeCost(std::size_t label1, std::size_t label2) const {
// 	return env_->ged_data_node_cost(label1, label2);
// }


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

}

#endif /* SRC_GEDLIB_BIND_IPP */

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