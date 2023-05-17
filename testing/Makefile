include ../make.inc

# print-%  : ; @echo $* = $($*)

# ROOTS
_KBLAS_ROOT_=..

# include and lib paths
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include -I${_KBLAS_ROOT_}/src
INCLUDES+= -I${_NLOPT_ROOT_}/include
INCLUDES+= -I${_GSL_ROOT_}/include

ifdef _USE_MAGMA_
	INCLUDES+= -I$(_MAGMA_ROOT_)/include
endif
ifdef _USE_MKL_
	INCLUDES+= -I${_MKL_ROOT_}/include
endif
ifdef _USE_OPENBLAS_
	INCLUDES+= -I${_OPENBLAS_INCLUDE_}
endif

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib
LIB_PATH+= -L${_NLOPT_ROOT_}/lib
LIB_PATH+= -L${_GSL_ROOT_}/lib

ifdef _USE_MAGMA_
	LIB_PATH+= -L${_MAGMA_ROOT_}/lib
endif
ifdef _USE_MKL_
	LIB_PATH+= -L${_MKL_ROOT_}/lib/intel64
endif
ifdef _USE_OPENBLAS_
	LIB_PATH+= -L${_OPENBLAS_LIB_}
endif

# libraries to link against
LIB= -lm -l${LIB_KBLAS_NAME}
LIB+= -lnlopt  -lgsl
ifdef _USE_MAGMA_
	LIB+= -lmagma -lcusparse
endif
LIB+= -lcublas -lcudart
ifdef _USE_MKL_
	# LIB+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl
	LIB+= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	# LIB+= -mkl=sequential
	# LIB+= -mkl=parallel
endif
ifdef _USE_OPENBLAS_
	LIB+= -lopenblas
endif
LIB+= -lgomp
LIB+= -lstdc++

KBLAS_LIB=../lib/lib${LIB_KBLAS_NAME}.a

INC_SRC=ckernel.h

OBJ_DIR=./obj
BIN_DIR=./bin


#$(info $$_SUPPORT_BATCH_TR_ is [${_SUPPORT_BATCH_TR_}])
ifdef _SUPPORT_BATCH_TR_
KBLAS_BATCH_DIR=src
include $(KBLAS_BATCH_DIR)/Makefile
endif

all: $(EXE_BATCH)

$(OBJ_DIR)/testing_helper.o: testing_helper.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INCLUDES) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/ckernel.o: ckernel.cpp
	$(CXX) $(COPTS) $(INCLUDES) -c $< -o $@

$(EXE_BATCH): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(KBLAS_LIB) $(OBJ_DIR)/testing_helper.o $(OBJ_DIR)/ckernel.o
	$(CC) $(COPTS) $(OBJ_DIR)/testing_helper.o $(OBJ_DIR)/ckernel.o $< -o $@ $(LIB_PATH) $(LIB)

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE_BATCH)
