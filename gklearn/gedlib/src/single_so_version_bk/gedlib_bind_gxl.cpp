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
 * @file gedlib_bind_gxl.cpp
 * @brief implementations of classes and functions to call easily GebLib in Python without Gedlib's types
 */
#pragma once
//#ifndef GEDLIBBIND_IPP
//#define GEDLIBBIND_IPP

//Include standard libraries + GedLib library
// #include <iostream>
// #include "GedLibBind.h"
// #include "../include/gedlib-master/src/env/ged_env.hpp"
//#include "../include/gedlib-master/median/src/median_graph_estimator.hpp"
#include "gedlib_bind_gxl.hpp"

using namespace std;

//Definition of types and templates used in this code for my human's memory :).
//ged::GEDEnv<UserNodeID, UserNodeLabel, UserEdgeLabel> env;
//template<class UserNodeID, class UserNodeLabel, class UserEdgeLabel> struct ExchangeGraph

//typedef std::map<std::string, std::string> GXLLabel;
//typedef std::string GXLNodeID;


namespace pyged {

PyGEDEnvGXL::PyGEDEnvGXL () {
	env_ = new ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	this->initialized = false;
}

PyGEDEnvGXL::~PyGEDEnvGXL () {
	if (env_ != NULL) {
		delete env_;
		env_ = NULL;
	}
}

// bool initialized = false; //Initialization boolean (because Env has one but not accessible).

bool PyGEDEnvGXL::isInitialized() {
	return initialized;
}

void PyGEDEnvGXL::restartEnv() {
	if (env_ != NULL) {
		delete env_;
		env_ = NULL;
	}
	env_ = new ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	initialized = false;
}

void PyGEDEnvGXL::loadGXLGraph(const std::string & pathFolder, const std::string & pathXML, bool node_type, bool edge_type) {
	 std::vector<ged::GEDGraph::GraphID> tmp_graph_ids(env_->load_gxl_graph(pathFolder, pathXML,
	 	(node_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
		(edge_type ? ged::Options::GXLNodeEdgeType::LABELED : ged::Options::GXLNodeEdgeType::UNLABELED),
		std::unordered_set<std::string>(), std::unordered_set<std::string>()));
}

std::pair<std::size_t,std::size_t> PyGEDEnvGXL::getGraphIds() const {
	return env_->graph_ids();
}

std::vector<std::size_t> PyGEDEnvGXL::getAllGraphIds() {
	std::vector<std::size_t> listID;
	for (std::size_t i = env_->graph_ids().first; i != env_->graph_ids().second; i++) {
		listID.push_back(i);
    }
	return listID;
}

const std::string PyGEDEnvGXL::getGraphClass(std::size_t id) const {
	return env_->get_graph_class(id);
}

const std::string PyGEDEnvGXL::getGraphName(std::size_t id) const {
	return env_->get_graph_name(id);
}

std::size_t PyGEDEnvGXL::addGraph(const std::string & graph_name, const std::string & graph_class) {
	ged::GEDGraph::GraphID newId = env_->add_graph(graph_name, graph_class);
	initialized = false;
	return std::stoi(std::to_string(newId));
}

void PyGEDEnvGXL::addNode(std::size_t graphId, const std::string & nodeId, const std::map<std::string, std::string> & nodeLabel) {
	env_->add_node(graphId, nodeId, nodeLabel);
	initialized = false;
}

/*void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel) {
	env_->add_edge(graphId, tail, head, edgeLabel);
}*/

void PyGEDEnvGXL::addEdge(std::size_t graphId, const std::string & tail, const std::string & head, const std::map<std::string, std::string> & edgeLabel, bool ignoreDuplicates) {
	env_->add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
	initialized = false;
}

void PyGEDEnvGXL::clearGraph(std::size_t graphId) {
	env_->clear_graph(graphId);
	initialized = false;
}

ged::ExchangeGraph<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> PyGEDEnvGXL::getGraph(std::size_t graphId) const {
	return env_->get_graph(graphId);
}

std::size_t PyGEDEnvGXL::getGraphInternalId(std::size_t graphId) {
	return getGraph(graphId).id;
}

std::size_t PyGEDEnvGXL::getGraphNumNodes(std::size_t graphId) {
	return getGraph(graphId).num_nodes;
}

std::size_t PyGEDEnvGXL::getGraphNumEdges(std::size_t graphId) {
	return getGraph(graphId).num_edges;
}

std::vector<std::string> PyGEDEnvGXL::getGraphOriginalNodeIds(std::size_t graphId) {
	return getGraph(graphId).original_node_ids;
}

std::vector<std::map<std::string, std::string>> PyGEDEnvGXL::getGraphNodeLabels(std::size_t graphId) {
	return getGraph(graphId).node_labels;
}

std::map<std::pair<std::size_t, std::size_t>, std::map<std::string, std::string>> PyGEDEnvGXL::getGraphEdges(std::size_t graphId) {
	return getGraph(graphId).edge_labels;
}

std::vector<std::vector<std::size_t>> PyGEDEnvGXL::getGraphAdjacenceMatrix(std::size_t graphId) {
	return getGraph(graphId).adj_matrix;
}

void PyGEDEnvGXL::setEditCost(std::string editCost, std::vector<double> editCostConstants) {
	env_->set_edit_costs(translateEditCost(editCost), editCostConstants);
}

void PyGEDEnvGXL::setPersonalEditCost(std::vector<double> editCostConstants) {
	//env_->set_edit_costs(Your EditCost Class(editCostConstants));
}

// void PyGEDEnvGXL::initEnv() {
// 	env_->init();
// 	initialized = true;
// }

void PyGEDEnvGXL::initEnv(std::string initOption, bool print_to_stdout) {
	env_->init(translateInitOptions(initOption), print_to_stdout);
	initialized = true;
}

void PyGEDEnvGXL::setMethod(std::string method, const std::string & options) {
	env_->set_method(translateMethod(method), options);
}

void PyGEDEnvGXL::initMethod() {
	env_->init_method();
}

double PyGEDEnvGXL::getInitime() const {
	return env_->get_init_time();
}

void PyGEDEnvGXL::runMethod(std::size_t g, std::size_t h) {
	env_->run_method(g, h);
}

double PyGEDEnvGXL::getUpperBound(std::size_t g, std::size_t h) const {
	return env_->get_upper_bound(g, h);
}

double PyGEDEnvGXL::getLowerBound(std::size_t g, std::size_t h) const {
	return env_->get_lower_bound(g, h);
}

std::vector<long unsigned int> PyGEDEnvGXL::getForwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_forward_map();
}

std::vector<long unsigned int> PyGEDEnvGXL::getBackwardMap(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).get_backward_map();
}

std::size_t PyGEDEnvGXL::getNodeImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).image(nodeId);
}

std::size_t PyGEDEnvGXL::getNodePreImage(std::size_t g, std::size_t h, std::size_t nodeId) const {
	return env_->get_node_map(g, h).pre_image(nodeId);
}

double PyGEDEnvGXL::getInducedCost(std::size_t g, std::size_t h) const {
	return env_->get_node_map(g, h).induced_cost();
}

std::vector<pair<std::size_t, std::size_t>> PyGEDEnvGXL::getNodeMap(std::size_t g, std::size_t h) {
	std::vector<pair<std::size_t, std::size_t>> res;
	std::vector<ged::NodeMap::Assignment> relation;
	env_->get_node_map(g, h).as_relation(relation);
	for (const auto & assignment : relation) {
		res.push_back(std::make_pair(assignment.first, assignment.second));
	}
	return res;
}

std::vector<std::vector<int>> PyGEDEnvGXL::getAssignmentMatrix(std::size_t g, std::size_t h) {
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

std::vector<std::vector<unsigned long int>> PyGEDEnvGXL::getAllMap(std::size_t g, std::size_t h) {
	std::vector<std::vector<unsigned long int>> res;
	res.push_back(getForwardMap(g, h));
	res.push_back(getBackwardMap(g,h));
	return res;
}

double PyGEDEnvGXL::getRuntime(std::size_t g, std::size_t h) const {
	return env_->get_runtime(g, h);
}

bool PyGEDEnvGXL::quasimetricCosts() const {
	return env_->quasimetric_costs();
}

std::vector<std::vector<size_t>> PyGEDEnvGXL::hungarianLSAP(std::vector<std::vector<std::size_t>> matrixCost) {
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

std::vector<std::vector<double>> PyGEDEnvGXL::hungarianLSAPE(std::vector<std::vector<double>> matrixCost) {
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

std::size_t PyGEDEnvGXL::getNumNodeLabels() const {
	return env_->num_node_labels();
}

std::map<std::string, std::string> PyGEDEnvGXL::getNodeLabel(std::size_t label_id) const {
	return env_->get_node_label(label_id);
}

std::size_t PyGEDEnvGXL::getNumEdgeLabels() const {
	return env_->num_edge_labels();
}

std::map<std::string, std::string> PyGEDEnvGXL::getEdgeLabel(std::size_t label_id) const {
	return env_->get_edge_label(label_id);
}

// std::size_t PyGEDEnvGXL::getNumNodes(std::size_t graph_id) const {
// 	return env_->get_num_nodes(graph_id);
// }

double PyGEDEnvGXL::getAvgNumNodes() const {
	return env_->get_avg_num_nodes();
}

double PyGEDEnvGXL::getNodeRelCost(const std::map<std::string, std::string> & node_label_1, const std::map<std::string, std::string> & node_label_2) const {
	return env_->node_rel_cost(node_label_1, node_label_2);
}

double PyGEDEnvGXL::getNodeDelCost(const std::map<std::string, std::string> & node_label) const {
	return env_->node_del_cost(node_label);
}

double PyGEDEnvGXL::getNodeInsCost(const std::map<std::string, std::string> & node_label) const {
	return env_->node_ins_cost(node_label);
}

std::map<std::string, std::string> PyGEDEnvGXL::getMedianNodeLabel(const std::vector<std::map<std::string, std::string>> & node_labels) const {
	return env_->median_node_label(node_labels);
}

double PyGEDEnvGXL::getEdgeRelCost(const std::map<std::string, std::string> & edge_label_1, const std::map<std::string, std::string> & edge_label_2) const {
	return env_->edge_rel_cost(edge_label_1, edge_label_2);
}

double PyGEDEnvGXL::getEdgeDelCost(const std::map<std::string, std::string> & edge_label) const {
	return env_->edge_del_cost(edge_label);
}

double PyGEDEnvGXL::getEdgeInsCost(const std::map<std::string, std::string> & edge_label) const {
	return env_->edge_ins_cost(edge_label);
}

std::map<std::string, std::string> PyGEDEnvGXL::getMedianEdgeLabel(const std::vector<std::map<std::string, std::string>> & edge_labels) const {
	return env_->median_edge_label(edge_labels);
}

std::string PyGEDEnvGXL::getInitType() const {
	return initOptionsToString(env_->get_init_type());
}

double PyGEDEnvGXL::computeInducedCost(std::size_t g_id, std::size_t h_id, std::vector<pair<std::size_t, std::size_t>> relation) const {
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




// double PyGEDEnvGXL::getNodeCost(std::size_t label1, std::size_t label2) const {
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

} // namespace pyged

//#endif /* SRC_GEDLIB_BIND_IPP */