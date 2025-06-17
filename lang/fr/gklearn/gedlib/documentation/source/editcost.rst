How to add your own editCost class 
=========================================

When you choose your cost function, you can decide some parameters to personalize the function. But if you have some graphs which its type doesn't correpond to the choices, you can create your edit cost function. 

For this, you have to write it in C++. 

C++ side
-------------

You class must inherit to EditCost class, which is an asbtract class. You can find it here : include/gedlib-master/src/edit_costs

You can inspire you to the others to understand how to use it. You have to override these functions : 

- virtual double node_ins_cost_fun(const UserNodeLabel & node_label) const final;
- virtual double node_del_cost_fun(const UserNodeLabel & node_label) const final;
- virtual double node_rel_cost_fun(const UserNodeLabel & node_label_1, const UserNodeLabel & node_label_2) const final;
- virtual double edge_ins_cost_fun(const UserEdgeLabel & edge_label) const final;
- virtual double edge_del_cost_fun(const UserEdgeLabel & edge_label) const final;
- virtual double edge_rel_cost_fun(const UserEdgeLabel & edge_label_1, const UserEdgeLabel & edge_label_2) const final;

You can add some attributes for parameters use or more functions, but these are unavoidable.

When your class is ready, please go to the C++ Bind here : src/GedLibBind.cpp . The function is :

	void setPersonalEditCost(std::vector<double> editCostConstants){env.set_edit_costs(Your EditCost Class(editCostConstants));}

You have just to initialize your class. Parameters aren't mandatory, empty by default. If your class doesn't have one, you can skip this. After that, you have to recompile the project. 

Python side
----------------

For this, use setup.py with this command in a linux shell::

  python3 setup.py build_ext --inplace

You can also make it in Python 2. 

Now you can use your edit cost function with the Python function set_personal_edit_cost(edit_cost_constant). 

If you want more informations on C++, you can check the documentation of the original library here : https://github.com/dbblumenthal/gedlib

