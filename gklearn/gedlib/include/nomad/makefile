UNAME := $(shell uname)

EXE                    = ../bin/nomad
EXE_MPI                = ../bin/nomad.MPI

LIB_DIR                = ../lib
LIB_NAME               = libnomad.so
LIB_NAME_MPI           = libnomad.MPI.so

LIB                    = $(LIB_DIR)/$(LIB_NAME)
LIB_MPI                = $(LIB_DIR)/$(LIB_NAME_MPI)

LIB_CURRENT_VERSION    = 3.8.1

SGTELIB_HOME           = ../ext/sgtelib
LIBSGTELIB_NAME        = libsgtelib.so

COMPILATOR             = g++
COMPILATOR_MPI         = mpic++

BUILD_DIR_NO_MPI       = ../builds/ObjsNoMPI 
BUILD_DIR_MPI          = ../builds/ObjsMPI
BUILD_DIR = 

CXXFLAGS	       = -O2 -Wall -fpic
# CXXFLAGS	       = -g -Wall -fpic
CXXFLAGS_LIBS          =
CXXFLAGS_EXE           =           
ifeq ($(UNAME), Darwin)
CXXFLAGS_LIBS          = -current_version $(LIB_CURRENT_VERSION) -compatibility_version $(LIB_CURRENT_VERSION) -install_name $(LIB_NAME) -headerpad_max_install_names
CXXFLAGS_LIBS_MPI      = -current_version $(LIB_CURRENT_VERSION) -compatibility_version $(LIB_CURRENT_VERSION) -install_name $(LIB_NAME_MPI) -headerpad_max_install_names
CXXFLAGS_EXE          += -headerpad_max_install_names
endif
ifeq ($(UNAME), Linux)
CXXFLAGS_LIBS         += -Wl,-soname,'$(LIB_NAME)' -Wl,-rpath-link,'.' -Wl,-rpath,'$$ORIGIN'
CXXFLAGS_LIBS_MPI     += -Wl,-soname,'$(LIB_NAME_MPI)' -Wl,-rpath-link,'.' -Wl,-rpath,'$$ORIGIN'
CXXFLAGS_EXE          += -Wl,-rpath-link,'../lib' -Wl,-rpath,'$$ORIGIN/../lib'
CXXFLAGS              += -ansi
endif

CXXFLAGS_MPI 	       = $(CXXFLAGS) -DUSE_MPI # -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX

LDLIBS_EXE             = -lm -lnomad 
LDLIBS_LIBS            = -lm -lsgtelib
LDLIBS_EXE_MPI         = -lm -lnomad.MPI -lmpi
LDLIBS_LIBS_MPI        = $(LDLIBS_LIBS) -lmpi

LDFLAGS		       = -L$(LIB_DIR)

INCLUDE                = -I. -I$(SGTELIB_HOME)/src

COMPILE                = 
COMPILE_NO_MPI         = $(COMPILATOR) $(CXXFLAGS) $(INCLUDE) -c
COMPILE_MPI            = $(COMPILATOR_MPI) $(CXXFLAGS_MPI) $(INCLUDE) -c

OBJS_LIB               = Barrier.o Cache.o Cache_File_Point.o Cache_Point.o \
                         Cache_Search.o Clock.o Direction.o Directions.o Display.o \
                         Double.o Eval_Point.o Evaluator.o Evaluator_Control.o \
                         Exception.o Extended_Poll.o L_Curve.o LH_Search.o Mads.o \
                         OrthogonalMesh.o Model_Sorted_Point.o Model_Stats.o Multi_Obj_Evaluator.o \
                         Parameters.o Parameter_Entries.o Parameter_Entry.o \
                         Pareto_Front.o Pareto_Point.o Phase_One_Evaluator.o \
                         Phase_One_Search.o Point.o Priority_Eval_Point.o Quad_Model.o \
                         Quad_Model_Evaluator.o Quad_Model_Search.o Random_Pickup.o \
                         RNG.o Sgtelib_Model_Evaluator.o Sgtelib_Model_Search.o Sgtelib_Model_Manager.o \
		         Signature.o Slave.o Speculative_Search.o Stats.o \
			 SMesh.o GMesh.o XMesh.o \
                         utils.o Variable_Group.o VNS_Search.o 
                         
OBJ_MAIN               = nomad.o 

OBJS_LIB              := $(addprefix $(BUILD_DIR)/,$(OBJS_LIB))
OBJ_MAIN              := $(addprefix $(BUILD_DIR)/,$(OBJ_MAIN))


define DisplayCommand
	@if $(1) ; then echo " ---> $@ OK" ; else echo " compilation failed for $@" ; fi
endef
define DisplayCommand0
	@if $(1) ; then echo " ----> OK" ; else echo " ----> FAILED" ; fi
endef

define Mess
	@echo $1 $2
endef

all:
	@echo
	@mkdir -p $(BUILD_DIR_NO_MPI)
	@echo " ===== Creating the SGTELIB shared objects library and executables ===== "
	@$(MAKE) -C $(SGTELIB_HOME)/src SHORT_DISPLAY='y' all
	@echo " ===== Copy $(LIBSGTELIB_NAME) into NOMAD lib dir ====="
	cp -f $(SGTELIB_HOME)/lib/$(LIBSGTELIB_NAME) $(LIB_DIR)/$(LIBSGTELIB_NAME) 
	@echo
	@$(MAKE) BUILD_DIR=$(BUILD_DIR_NO_MPI) COMPILE='$(COMPILE_NO_MPI)' $(LIB) $(EXE)
	
mpi:
	@mkdir -p $(BUILD_DIR_MPI)
	@echo " ===== Creating the SGTELIB shared objects library ===== "
	@$(MAKE) -C $(SGTELIB_HOME)/src SHORT_DISPLAY='y' all
	@echo " ===== Copy $(LIBSGTELIB_NAME) into NOMAD lib dir ====="
	@cp -f $(SGTELIB_HOME)/lib/$(LIBSGTELIB_NAME) $(LIB_DIR)/$(LIBSGTELIB_NAME)
	@echo
	@$(MAKE) BUILD_DIR=$(BUILD_DIR_MPI) COMPILE='$(COMPILE_MPI)' $(LIB_MPI) $(EXE_MPI)

installAllNomad: all mpi
	@echo
	@echo " ===== cleaning obj files and build directories ===== "
	$(call DisplayCommand0,rm -rf $(BUILD_DIR_NO_MPI) $(BUILD_DIR_MPI))
	@echo
	
clean:
	@echo
	@echo " ===== Cleaning obj files and build directories ===== "
	@echo " == Cleaning SGTELIB =="
	@$(MAKE) -C $(SGTELIB_HOME)/src SHORT_DISPLAY='y' clean
	@echo " == Cleaning NOMAD =="  
	$(call DisplayCommand0,rm -rf $(BUILD_DIR_NO_MPI) $(BUILD_DIR_MPI))
	@echo
	
del: clean
	@echo
	@echo " ===== Deleting files ===== "
	@echo " == del SGTELIB =="
	@$(MAKE) -C $(SGTELIB_HOME)/src SHORT_DISPLAY='y' del
	@echo " == del NOMAD == "
	@echo "   --> cleaning trash files " 
	$(call DisplayCommand0,rm -f core *~)
	@echo "   --> cleaning exe file" 
	$(call DisplayCommand0,rm -f $(EXE) $(EXE_MPI))
	@echo "   --> cleaning lib files"
	$(call DisplayCommand0,rm -f $(LIB) $(LIB_DIR)/$(LIBSGTELIB_NAME) $(LIB_MPI))
	@echo	

$(EXE): $(LIB) mess $(OBJ_MAIN)
	@echo
	@echo " ===== Creating the NOMAD executable release version (no MPI) ===== "
	$(call DisplayCommand,$(COMPILATOR) -o $(EXE) $(OBJ_MAIN) $(LDLIBS_EXE) $(LDFLAGS) $(CXXFLAGS_EXE))
ifeq ($(UNAME), Darwin)
	@install_name_tool -change $(LIB_NAME) @loader_path/$(LIB) $(EXE)
endif
	@echo

.PHONY:mess
mess: 
	@echo
	@echo " ===== Compiling Nomad objects ===== "

$(LIB): mess $(OBJS_LIB) $(OBJ_MAIN)
	@mkdir -p $(LIB_DIR)
	@echo 	
	@echo " ===== Creating the NOMAD shared objects library (no MPI) ===== " 
	@rm -f $(LIB)
	$(call DisplayCommand,$(COMPILATOR) -shared -o $(LIB) $(OBJS_LIB) $(LDLIBS_LIBS) $(LDFLAGS) $(CXXFLAGS_LIBS))
ifeq ($(UNAME), Darwin)
	@install_name_tool -change $(LIBSGTELIB_NAME) @loader_path/$(LIBSGTELIB_NAME) $(LIB)
endif
	@echo
 
$(EXE_MPI): $(LIB_MPI) $(OBJ_MAIN)
	@echo
	@echo " ===== Creating the NOMAD executable release version (MPI) ===== "
	$(call DisplayCommand,$(COMPILATOR_MPI) -o $(EXE_MPI) $(OBJ_MAIN) $(LDLIBS_EXE_MPI) $(LDFLAGS) $(CXXFLAGS_EXE))
ifeq ($(UNAME), Darwin)
	@install_name_tool -change $(LIB_NAME_MPI) @loader_path/$(LIB_MPI) $(EXE_MPI)
endif
	@echo


$(LIB_MPI): mess $(OBJS_LIB)
	@mkdir -p $(LIB_DIR)
	@echo
	@echo " ===== Creating the NOMAD shared objects library (MPI) ===== "
	@rm -f $(LIB_MPI)
	$(call DisplayCommand,$(COMPILATOR_MPI) -shared -o $(LIB_MPI) $(OBJS_LIB) $(LDLIBS_LIBS_MPI) $(LDFLAGS) $(CXXFLAGS_LIBS_MPI))
ifeq ($(UNAME), Darwin)
	@install_name_tool -change $(LIBSGTELIB_NAME) @loader_path/$(LIBSGTELIB_NAME) $(LIB_MPI)
endif
	@echo

	
#$(BUILD_DIR)/%.o: %.cpp %.hpp
#	$(COMPILE) $< -o $@
	
$(BUILD_DIR)/Barrier.o: Barrier.cpp Barrier.hpp Filter_Point.hpp Set_Element.hpp
	$(call DisplayCommand,$(COMPILE) Barrier.cpp -o $@)

$(BUILD_DIR)/Cache.o: Cache.cpp Cache.hpp
	$(call DisplayCommand,$(COMPILE) Cache.cpp -o $@)

$(BUILD_DIR)/Cache_File_Point.o: Cache_File_Point.cpp Cache_File_Point.hpp \
                    Eval_Point.cpp Eval_Point.hpp Uncopyable.hpp
	$(call DisplayCommand,$(COMPILE) Cache_File_Point.cpp -o $@)

$(BUILD_DIR)/Cache_Point.o: Cache_Point.cpp Cache_Point.hpp
	$(call DisplayCommand,$(COMPILE) Cache_Point.cpp -o $@)

$(BUILD_DIR)/Cache_Search.o: Cache_Search.cpp Cache_Search.hpp Search.hpp
	$(call DisplayCommand,$(COMPILE) Cache_Search.cpp -o $@)

$(BUILD_DIR)/Clock.o: Clock.cpp Clock.hpp
	$(call DisplayCommand,$(COMPILE) Clock.cpp -o $@)

$(BUILD_DIR)/Direction.o: Direction.cpp Direction.hpp Point.hpp
	$(call DisplayCommand,$(COMPILE) Direction.cpp -o $@)

$(BUILD_DIR)/Directions.o: Directions.cpp Directions.hpp OrthogonalMesh.hpp Random_Pickup.hpp RNG.hpp
	$(call DisplayCommand,$(COMPILE) Directions.cpp -o $@)

$(BUILD_DIR)/Display.o: Display.cpp Display.hpp utils.hpp
	$(call DisplayCommand,$(COMPILE) Display.cpp -o $@)

$(BUILD_DIR)/Double.o: Double.cpp Double.hpp Exception.hpp Display.hpp
	$(call DisplayCommand,$(COMPILE) Double.cpp -o $@)

$(BUILD_DIR)/Eval_Point.o: Eval_Point.cpp Eval_Point.hpp Parameters.hpp Cache_File_Point.hpp \
              Set_Element.hpp
	$(call DisplayCommand,$(COMPILE) Eval_Point.cpp -o $@)

$(BUILD_DIR)/Evaluator.o: Evaluator.cpp Evaluator.hpp Priority_Eval_Point.hpp Stats.hpp
	$(call DisplayCommand,$(COMPILE) Evaluator.cpp -o $@)

$(BUILD_DIR)/Evaluator_Control.o: Evaluator_Control.cpp Evaluator_Control.hpp \
            Barrier.hpp Pareto_Front.hpp Slave.hpp Quad_Model.hpp        
	$(call DisplayCommand,$(COMPILE) Evaluator_Control.cpp -o $@)

$(BUILD_DIR)/Exception.o: Exception.cpp Exception.hpp
	$(call DisplayCommand,$(COMPILE) Exception.cpp -o $@)

$(BUILD_DIR)/Extended_Poll.o: Extended_Poll.cpp Extended_Poll.hpp Signature_Element.hpp \
                 Set_Element.hpp Mads.hpp
	$(call DisplayCommand,$(COMPILE) Extended_Poll.cpp -o $@)

$(BUILD_DIR)/L_Curve.o: L_Curve.cpp L_Curve.hpp Double.hpp Uncopyable.hpp
	$(call DisplayCommand,$(COMPILE) L_Curve.cpp -o $@)

$(BUILD_DIR)/LH_Search.o: LH_Search.cpp LH_Search.hpp Search.hpp Mads.hpp RNG.hpp Evaluator_Control.hpp
	$(call DisplayCommand,$(COMPILE) LH_Search.cpp -o $@)

$(BUILD_DIR)/Mads.o: Mads.cpp Mads.hpp Evaluator_Control.hpp L_Curve.hpp \
        LH_Search.hpp LH_Search.cpp \
        Speculative_Search.cpp Speculative_Search.hpp \
        Extended_Poll.cpp Extended_Poll.hpp \
        VNS_Search.hpp VNS_Search.cpp \
        Quad_Model_Search.hpp Quad_Model_Search.cpp \
        Cache_Search.hpp Cache_Search.cpp \
        Phase_One_Search.cpp Phase_One_Search.hpp
	$(call DisplayCommand,$(COMPILE) Mads.cpp -o $@)

$(BUILD_DIR)/OrthogonalMesh.o: OrthogonalMesh.hpp
	$(call DisplayCommand,$(COMPILE) OrthogonalMesh.cpp -o $@)

$(BUILD_DIR)/GMesh.o: GMesh.cpp GMesh.hpp OrthogonalMesh.cpp OrthogonalMesh.hpp
	$(call DisplayCommand,$(COMPILE) GMesh.cpp -o $@)

$(BUILD_DIR)/SMesh.o: SMesh.cpp SMesh.hpp OrthogonalMesh.cpp OrthogonalMesh.hpp
	$(call DisplayCommand,$(COMPILE) SMesh.cpp -o $@)

$(BUILD_DIR)/XMesh.o: XMesh.cpp XMesh.hpp OrthogonalMesh.cpp OrthogonalMesh.hpp
	$(call DisplayCommand,$(COMPILE) XMesh.cpp -o $@)

$(BUILD_DIR)/Multi_Obj_Evaluator.o: Multi_Obj_Evaluator.cpp Multi_Obj_Evaluator.hpp Phase_One_Evaluator.hpp
	$(call DisplayCommand,$(COMPILE) Multi_Obj_Evaluator.cpp -o $@)

$(BUILD_DIR)/Model_Sorted_Point.o: Model_Sorted_Point.cpp Model_Sorted_Point.hpp Point.hpp
	$(call DisplayCommand,$(COMPILE) Model_Sorted_Point.cpp -o $@)

$(BUILD_DIR)/Model_Stats.o: Model_Stats.cpp Model_Stats.hpp Double.hpp
	$(call DisplayCommand,$(COMPILE) Model_Stats.cpp -o $@)

$(BUILD_DIR)/nomad.o: nomad.cpp nomad.hpp  Mads.hpp
	$(call DisplayCommand,$(COMPILE) nomad.cpp -o $@)

$(BUILD_DIR)/Parameters.o: Parameters.cpp Parameters.hpp  Parameter_Entries.hpp Signature.hpp
	$(call DisplayCommand,$(COMPILE) Parameters.cpp -o $@)

$(BUILD_DIR)/Parameter_Entries.o: Parameter_Entries.cpp Parameter_Entries.hpp Parameter_Entry.hpp
	$(call DisplayCommand,$(COMPILE) Parameter_Entries.cpp -o $@)

$(BUILD_DIR)/Parameter_Entry.o: Parameter_Entry.hpp Parameter_Entry.cpp  Display.hpp Uncopyable.hpp
	$(call DisplayCommand,$(COMPILE) Parameter_Entry.cpp -o $@)

$(BUILD_DIR)/Pareto_Front.o: Pareto_Front.cpp Pareto_Front.hpp Pareto_Point.hpp
	$(call DisplayCommand,$(COMPILE) Pareto_Front.cpp -o $@)

$(BUILD_DIR)/Pareto_Point.o: Pareto_Point.cpp Pareto_Point.hpp Multi_Obj_Evaluator.hpp
	$(call DisplayCommand,$(COMPILE) Pareto_Point.cpp -o $@)

$(BUILD_DIR)/Phase_One_Evaluator.o: Phase_One_Evaluator.cpp Phase_One_Evaluator.hpp Evaluator.hpp
	$(call DisplayCommand,$(COMPILE) Phase_One_Evaluator.cpp -o $@)

$(BUILD_DIR)/Phase_One_Search.o: Phase_One_Search.cpp Phase_One_Search.hpp Mads.hpp \
                    Search.hpp Evaluator_Control.hpp
	$(call DisplayCommand,$(COMPILE) Phase_One_Search.cpp -o $@)

$(BUILD_DIR)/Point.o: Point.cpp Point.hpp Double.hpp
	$(call DisplayCommand,$(COMPILE) Point.cpp -o $@)

$(BUILD_DIR)/Priority_Eval_Point.o: Priority_Eval_Point.cpp Priority_Eval_Point.hpp Eval_Point.hpp \
                       Set_Element.hpp
	$(call DisplayCommand,$(COMPILE) Priority_Eval_Point.cpp -o $@)

$(BUILD_DIR)/Quad_Model.o: Quad_Model.cpp Quad_Model.hpp Cache.hpp Model_Sorted_Point.hpp
	$(call DisplayCommand,$(COMPILE) Quad_Model.cpp -o $@)

$(BUILD_DIR)/Quad_Model_Evaluator.o: Quad_Model_Evaluator.cpp Quad_Model_Evaluator.hpp \
                        Evaluator.hpp Search.hpp
	$(call DisplayCommand,$(COMPILE) Quad_Model_Evaluator.cpp -o $@)

$(BUILD_DIR)/Quad_Model_Search.o: Quad_Model_Search.cpp Quad_Model_Search.hpp Mads.hpp \
                     Quad_Model_Evaluator.hpp
	$(call DisplayCommand,$(COMPILE) Quad_Model_Search.cpp -o $@)

$(BUILD_DIR)/Random_Pickup.o: Random_Pickup.cpp Random_Pickup.hpp RNG.cpp RNG.hpp Uncopyable.hpp
	$(call DisplayCommand,$(COMPILE) Random_Pickup.cpp -o $@)

$(BUILD_DIR)/RNG.o: RNG.cpp RNG.hpp defines.hpp
	$(call DisplayCommand,$(COMPILE) RNG.cpp -o $@)

$(BUILD_DIR)/Signature.o: Signature.cpp Signature.hpp Variable_Group.hpp
	$(call DisplayCommand,$(COMPILE) Signature.cpp -o $@)

$(BUILD_DIR)/Slave.o: Slave.cpp Slave.hpp Evaluator.hpp
	$(call DisplayCommand,$(COMPILE) Slave.cpp -o $@)

$(BUILD_DIR)/Speculative_Search.o: Speculative_Search.cpp Speculative_Search.hpp Mads.hpp Search.hpp \
                      Evaluator_Control.hpp
	$(call DisplayCommand,$(COMPILE) Speculative_Search.cpp -o $@)

$(BUILD_DIR)/Stats.o: Stats.cpp Stats.hpp  Clock.hpp Double.hpp Model_Stats.hpp
	$(call DisplayCommand,$(COMPILE) Stats.cpp -o $@)

$(BUILD_DIR)/utils.o: utils.cpp utils.hpp defines.hpp 
	$(call DisplayCommand,$(COMPILE) utils.cpp -o $@)

$(BUILD_DIR)/Variable_Group.o: Variable_Group.cpp Variable_Group.hpp  Directions.hpp
	$(call DisplayCommand,$(COMPILE) Variable_Group.cpp -o $@)

$(BUILD_DIR)/VNS_Search.o: VNS_Search.cpp VNS_Search.hpp Search.hpp Evaluator_Control.hpp
	$(call DisplayCommand,$(COMPILE) VNS_Search.cpp -o $@)

$(BUILD_DIR)/Sgtelib_Model_Evaluator.o: Sgtelib_Model_Evaluator.cpp Sgtelib_Model_Evaluator.hpp Sgtelib_Model_Manager.hpp Search.hpp Evaluator_Control.hpp 
	$(call DisplayCommand,$(COMPILE) Sgtelib_Model_Evaluator.cpp -o $@)

$(BUILD_DIR)/Sgtelib_Model_Manager.o: Sgtelib_Model_Manager.cpp Sgtelib_Model_Evaluator.hpp Sgtelib_Model_Manager.hpp Parameters.hpp Search.hpp Evaluator_Control.hpp 
	$(call DisplayCommand,$(COMPILE) Sgtelib_Model_Manager.cpp  -o $@)

$(BUILD_DIR)/Sgtelib_Model_Search.o: Sgtelib_Model_Search.cpp Sgtelib_Model_Evaluator.hpp Sgtelib_Model_Manager.hpp Search.hpp LH_Search.hpp Evaluator_Control.hpp 
	$(call DisplayCommand,$(COMPILE) Sgtelib_Model_Search.cpp -o $@)
