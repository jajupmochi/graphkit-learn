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
 * @file GedLibBind.hpp
 * @brief Classe and function declarations to call easly GebLib in Python without Gedlib's types
 */
#ifndef GEDLIBBIND_HPP
#define GEDLIBBIND_HPP
 
//Include standard libraries.
#include <string>
#include <vector>
#include <map>
#include <list>
#include <iostream>
#include "../include/gedlib-master/src/env/ged_env.hpp"
#include "../include/gedlib-master/src/env/node_map.hpp"


/*!
 * @namespace pyged
 * @brief Global namespace for gedlibpy.
 */
namespace pyged {

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
* @brief Provides the API of GEDLIB for Python.
*/
class PyGEDEnv {

public:

	/*!
	 * @brief Constructor.
	 */
    PyGEDEnv();

    // PyGEDEnv();

	/*!
	 * @brief Destructor.
	 */
    ~PyGEDEnv();

    /*!
    * @brief Tests if the environment is initialized or not. 
    * @return Boolean @p true if the environment is initialized and @p false otherwise.
    */
    bool isInitialized();

    /*!
    * @brief Restart the environment (recall a new empty environment).
    */
    void restartEnv();

    /*!
    * @brief Loads graph given in the [GXL file format](http://www.gupro.de/GXL/).
    * @param[in] pathFolder The path to the directory containing the graphs.
    * @param[in] pathXML The path to a XML file thats lists the graphs contained in @p pathFolder that should be loaded.
    * @param[in] node_type Select if nodes are labeled or unlabeled.
    * @param[in] edge_type Select if edges are labeled or unlabeled.
    */
    void loadGXLGraph(const std::string & pathFolder, const std::string & pathXML, bool node_type, bool edge_type);

    /*!
    * @brief Provides access to the IDs of the graphs contained in the environment.
    * @return Pair <tt>(ID of first graphs, ID of last graph + 1)</tt> of graph IDs.
    * If both entries equal 0, the environment does not contain any graphs.
    */
    std::pair<std::size_t,std::size_t> getGraphIds() const;

    /*!
    * @brief Returns the list of graphs IDs which are loaded in the environment. 
    * @return A vector which contains all the graphs Ids. 
    */
    std::vector<std::size_t> getAllGraphIds();

    /*!
    * @brief Returns the graph class.
    * @param[in] id ID of an input graph that has been added to the environment.
    * @return Class of the input graph.
    */
    const std::string getGraphClass(std::size_t id) const;

    /*!
    * @brief Returns the graph name.
    * @param[in] id ID of an input graph that has been added to the environment.
    * @return Name of the input graph.
    */
    const std::string getGraphName(std::size_t id) const;

    /*!
    * @brief Adds a new uninitialized graph to the environment. Call initEnv() after calling this method.
    * @param[in] name The name of the added graph. Empty if not specified.
    * @param[in] class The class of the added graph. Empty if not specified.
    * @return The ID of the newly added graph.
    */
    std::size_t addGraph(const std::string & graph_name, const std::string & graph_class);

    /*!
    * @brief Adds a labeled node.
    * @param[in] graphId ID of graph that has been added to the environment.
    * @param[in] nodeId The user-specific ID of the vertex that has to be added.
    * @param[in] nodeLabel The label of the vertex that has to be added.
    */
    void addNode(std::size_t graphId, const std::string & nodeId, const std::map<std::string, std::string> & nodeLabel);

    /*!
    * @brief Adds a labeled edge.
    * @param[in] graphId ID of graph that has been added to the environment.
    * @param[in] tail The user-specific ID of the tail of the edge that has to be added.
    * @param[in] head The user-specific ID of the head of the edge that has to be added.
    * @param[in] edgeLabel The label of the vertex that has to be added. 
    * @param[in] ignoreDuplicates If @p true, duplicate edges are ignores. Otherwise, an exception is thrown if an existing edge is added to the graph.
    */
    void addEdge(std::size_t graphId, const std::string & tail, const std::string & head, const std::map<std::string, std::string> & edgeLabel, bool ignoreDuplicates = true);

    /*!
    * @brief Clears and de-initializes a graph that has previously been added to the environment. Call initEnv() after calling this method.
    * @param[in] graphId ID of graph that has to be cleared.
    */
    void clearGraph(std::size_t graphId);

    /*!
    * @brief Returns ged::ExchangeGraph representation.
    * @param graphId ID of the selected graph.
    * @return ged::ExchangeGraph representation of the selected graph.
    */
    ged::ExchangeGraph<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> getGraph(std::size_t graphId) const;

    /*!
    * @brief Returns the internal Id of a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The internal ID of the selected graph
    */
    std::size_t getGraphInternalId(std::size_t graphId);

    /*!
    * @brief Returns all the number of nodes on a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The number of nodes on the selected graph
    */
    std::size_t getGraphNumNodes(std::size_t graphId);

    /*!
    * @brief Returns all the number of edges on a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The number of edges on the selected graph
    */
    std::size_t getGraphNumEdges(std::size_t graphId);

    /*!
    * @brief Returns all th Ids of nodes on a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The list of IDs's nodes on the selected graph
    */
    std::vector<std::string> getGraphOriginalNodeIds(std::size_t graphId);

    /*!
    * @brief Returns all the labels of nodes on a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The list of labels's nodes on the selected graph
    */
    std::vector<std::map<std::string, std::string>> getGraphNodeLabels(std::size_t graphId);

    /*!
    * @brief Returns all the edges on a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The list of edges on the selected graph
    */
    std::map<std::pair<std::size_t, std::size_t>, std::map<std::string, std::string>> getGraphEdges(std::size_t graphId);

    /*!
    * @brief Returns the adjacence list of a graph, selected by its ID.
    * @param[in] graphId ID of an input graph that has been added to the environment.
    * @return The adjacence list of the selected graph
    */
    std::vector<std::vector<std::size_t>> getGraphAdjacenceMatrix(std::size_t graphId);

    /*!
    * @brief Sets the edit costs to one of the predefined edit costs.
    * @param[in] editCost Select one of the predefined edit costs.
    * @param[in] editCostConstants Parameters for the edit cost, empty by default.
    */
    void setEditCost(std::string editCost, std::vector<double> editCostConstants = {});

    /*!
    * @brief Sets the edit costs to a personal Edit Cost Class.
    * @param[in] editCostConstants Parameters for the edit cost, empty by default.
    * @note You have to add your class, which should inherit from EditCost class, in the function. After that, you can compile and use it in Python
    */
    void setPersonalEditCost(std::vector<double> editCostConstants = {});

    /*!
    * @brief Initializes the environment.
    * @param[in] initOption Select initialization options.
	* @param[in] print_to_stdout If set to @p true, the progress of the initialization is printed to std::out.
    */
    void initEnv(std::string initOption = "EAGER_WITH_SHUFFLED_COPIES", bool print_to_stdout = false);

    /*!
    * @brief Sets the GEDMethod to be used by run_method().
    * @param[in] method Select the method that is to be used.
    * @param[in] options An options string of the form @"[--@<option@> @<arg@>] [...]@" passed to the selected method.
    */
    void setMethod(std::string method, const std::string & options);

    /*!
    * @brief Initializes the method specified by call to set_method().
    */
    void initMethod();

    /*!
    * @brief Returns initialization time.
    * @return Runtime of the last call to init_method().
    */
    double getInitime() const;

    /*!
    * @brief Runs the GED method specified by call to set_method() between the graphs with IDs @p g and @p h.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    */
    void runMethod(std::size_t g, std::size_t h);

    /*!
    * @brief Returns upper bound for edit distance between the input graphs.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return Upper bound computed by the last call to run_method() with arguments @p g and @p h.
    */
    double getUpperBound(std::size_t g, std::size_t h) const;

    /*!
    * @brief Returns lower bound for edit distance between the input graphs.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return Lower bound computed by the last call to run_method() with arguments @p g and @p h.
    */
    double getLowerBound(std::size_t g,std::size_t h) const;

    /*!
    * @brief  Returns the forward map between nodes of the two indicated graphs. 
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return The forward map to the adjacence matrix computed by the last call to run_method() with arguments @p g and @p h.
    */
    std::vector<long unsigned int> getForwardMap(std::size_t g, std::size_t h) const;

    /*!
    * @brief  Returns the backward map between nodes of the two indicated graphs. 
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return The backward map to the adjacence matrix computed by the last call to run_method() with arguments @p g and @p h.
    */
    std::vector<long unsigned int> getBackwardMap(std::size_t g, std::size_t h) const;

    /*!
    * @brief Returns image of a node.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @param[in] nodeId Node whose image is to be returned.
    * @return Node to which node @p node is assigned.
    */
    std::size_t getNodeImage(std::size_t g, std::size_t h, std::size_t nodeId) const;

    /*!
    * @brief Returns pre-image of a node.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @param[in] nodeId Node whose pre-image is to be returned.
    * @return Node to which node @p node is assigned.
    */
    std::size_t getNodePreImage(std::size_t g, std::size_t h, std::size_t nodeId) const;

    /*!
    * @brief Returns the induced cost between the two indicated graphs.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return The induced cost between the two indicated graphs.
    */
    double getInducedCost(std::size_t g, std::size_t h) const;
    

    /*!
    * @brief Returns node map between the input graphs. This function duplicates datas. 
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return Node map computed by the last call to run_method() with arguments @p g and @p h.
    */
    std::vector<std::pair<std::size_t, std::size_t>> getNodeMap(std::size_t g, std::size_t h);

    /*!
    * @brief Returns assignment matrix between the input graphs. This function duplicates datas. 
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return Assignment matrix computed by the last call to run_method() with arguments @p g and @p h.
    */
    std::vector<std::vector<int>> getAssignmentMatrix(std::size_t g, std::size_t h);

    /*!
    * @brief  Returns a vector which contains the forward and the backward maps between nodes of the two indicated graphs. 
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return The forward and backward maps to the adjacence matrix computed by the last call to run_method() with arguments @p g and @p h.
    */
    std::vector<std::vector<unsigned long int>> getAllMap(std::size_t g, std::size_t h);

    /*!
    * @brief Returns runtime.
    * @param[in] g ID of an input graph that has been added to the environment.
    * @param[in] h ID of an input graph that has been added to the environment.
    * @return Runtime of last call to run_method() with arguments @p g and @p h.
    */
    double getRuntime(std::size_t g, std::size_t h) const;

    /*!
    * @brief Checks if the edit costs are quasimetric.
    * @return Boolean @p true if the edit costs are quasimetric and @p false, otherwise.
    */
    bool quasimetricCosts() const;

    /*!
    * @brief Applies the hungarian algorithm (LSAP) to a matrix cost.
    * @param[in] matrixCost The matrix cost.
    * @return the values of rho, varrho, u and v, in this order.
    */
    std::vector<std::vector<size_t>> hungarianLSAP(std::vector<std::vector<std::size_t>> matrixCost);

    /*!
    * @brief Applies the hungarian algorithm (LSAPE) to a matrix cost.
    * @param[in] matrixCost The matrix cost.
    * @return the values of rho, varrho, u and v, in this order.
    */
    std::vector<std::vector<double>> hungarianLSAPE(std::vector<std::vector<double>> matrixCost);

    /*!
	 * @brief Returns the number of node labels.
	 * @return Number of pairwise different node labels contained in the environment.
	 * @note If @p 1 is returned, the nodes are unlabeled.
	 */
	std::size_t getNumNodeLabels() const;

    /*!
	 * @brief Returns node label.
	 * @param[in] label_id ID of node label that should be returned. Must be between 1 and num_node_labels().
	 * @return Node label for selected label ID.
	 */
	std::map<std::string, std::string> getNodeLabel(std::size_t label_id) const;

    /*!
	 * @brief Returns the number of edge labels.
	 * @return Number of pairwise different edge labels contained in the environment.
	 * @note If @p 1 is returned, the edges are unlabeled.
	 */
	std::size_t getNumEdgeLabels() const;

	/*!
	 * @brief Returns edge label.
	 * @param[in] label_id ID of edge label that should be returned. Must be between 1 and num_node_labels().
	 * @return Edge label for selected label ID.
	 */
	std::map<std::string, std::string> getEdgeLabel(std::size_t label_id) const;

	// /*!
	//  * @brief Returns the number of nodes.
	//  * @param[in] graph_id ID of an input graph that has been added to the environment.
	//  * @return Number of nodes in the graph.
	//  */
	// std::size_t getNumNodes(std::size_t graph_id) const;

	/*!
	 * @brief Returns average number of nodes.
	 * @return Average number of nodes of the graphs contained in the environment.
	 */
	double getAvgNumNodes() const;

    /*!
	 * @brief Returns node relabeling cost.
	 * @param[in] node_label_1 First node label.
	 * @param[in] node_label_2 Second node label.
	 * @return Node relabeling cost for the given node labels.
	 */
	double getNodeRelCost(const std::map<std::string, std::string> & node_label_1, const std::map<std::string, std::string> & node_label_2) const;

	/*!
	 * @brief Returns node deletion cost.
	 * @param[in] node_label Node label.
	 * @return Cost of deleting node with given label.
	 */
	double getNodeDelCost(const std::map<std::string, std::string> & node_label) const;

	/*!
	 * @brief Returns node insertion cost.
	 * @param[in] node_label Node label.
	 * @return Cost of inserting node with given label.
	 */
	double getNodeInsCost(const std::map<std::string, std::string> & node_label) const;

	/*!
	 * @brief Computes median node label.
	 * @param[in] node_labels The node labels whose median should be computed.
	 * @return Median of the given node labels.
	 */
	std::map<std::string, std::string> getMedianNodeLabel(const std::vector<std::map<std::string, std::string>> & node_labels) const;

	/*!
	 * @brief Returns edge relabeling cost.
	 * @param[in] edge_label_1 First edge label.
	 * @param[in] edge_label_2 Second edge label.
	 * @return Edge relabeling cost for the given edge labels.
	 */
	double getEdgeRelCost(const std::map<std::string, std::string> & edge_label_1, const std::map<std::string, std::string> & edge_label_2) const;

	/*!
	 * @brief Returns edge deletion cost.
	 * @param[in] edge_label Edge label.
	 * @return Cost of deleting edge with given label.
	 */
	double getEdgeDelCost(const std::map<std::string, std::string> & edge_label) const;

	/*!
	 * @brief Returns edge insertion cost.
	 * @param[in] edge_label Edge label.
	 * @return Cost of inserting edge with given label.
	 */
	double getEdgeInsCost(const std::map<std::string, std::string> & edge_label) const;

	/*!
	 * @brief Computes median edge label.
	 * @param[in] edge_labels The edge labels whose median should be computed.
	 * @return Median of the given edge labels.
	 */
	std::map<std::string, std::string> getMedianEdgeLabel(const std::vector<std::map<std::string, std::string>> & edge_labels) const;

    /*!
	 * @brief Returns the initialization type of the last initialization.
	 * @return Initialization type in string.
	 */
	std::string getInitType() const;

    /*!
	 * @brief Computes the edit cost between two graphs induced by a node map.
	 * @param[in] g_id ID of input graph.
	 * @param[in] h_id ID of input graph.
     * @return Computed induced cost.
	 */
	double computeInducedCost(std::size_t g_id, std::size_t h_id, std::vector<pair<std::size_t, std::size_t>> relation) const;

    // /*!
	//  * @brief Returns node relabeling, insertion, or deletion cost.
	//  * @param[in] label1 First node label.
	//  * @param[in] label2 Second node label.
	//  * @return Node relabeling cost if @p label1 and @p label2 are both different from ged::dummy_label(),
	//  * node insertion cost if @p label1 equals ged::dummy_label and @p label2 does not,
	//  * node deletion cost if @p label1 does not equal ged::dummy_label and @p label2 does,
	//  * and 0 otherwise.
	//  */
	// double getNodeCost(std::size_t label1, std::size_t label2) const;


private:

    ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> * env_; // environment variable

    bool initialized; // initialization boolean (because env has one but not accessible)

};

}

#include "GedLibBind.ipp"

#endif /* SRC_GEDLIB_BIND_HPP */











// namespace shapes {
//     class Rectangle {
//         public:
//             int x0, y0, x1, y1;
//             Rectangle();
//             Rectangle(int x0, int y0, int x1, int y1);
//             ~Rectangle();
//             int getArea();
//             void getSize(int* width, int* height);
//             void move(int dx, int dy);
//     };
// }