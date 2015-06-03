#ARCH ?= #k100#geocluster #gpupc1 #D
#USE_AIVLIB_MODEL ?= 1
#MPI_ON ?= 1

ifdef MPI_ON
ifeq ($(ARCH),k100)
GCC  ?= /usr/mpi/gcc/openmpi-1.4.2-qlc/bin/mpicc
else
GCC  ?= mpic++
endif
else
GCC  ?= g++
endif
#NVCC := /home/zakirov/progs/cuda-7.0/bin/nvcc -ccbin $(GCC)

ifeq ($(ARCH),k100)
NVCC := /common/cuda-6.5/bin/nvcc -ccbin $(GCC) -O3 -G -g 
GENCODE_SM := -arch=sm_20
else ifeq ($(ARCH),geocluster)
NVCC := nvcc -ccbin $(GCC) -O3
GENCODE_SM := -arch=sm_50
else ifeq ($(ARCH),gpupc1)
NVCC := nvcc -ccbin $(GCC) -O3
GENCODE_SM := -arch=sm_35
else ifeq ($(ARCH),D)
NVCC := nvcc -ccbin $(GCC) -O3 
GENCODE_SM := -arch=sm_35
else
NVCC := nvcc -ccbin $(GCC) -O3
GENCODE_SM := -arch=sm_50
endif 
#NTIME = 0
ALL_DEFS := NS NA NV NTIME DYSH MPI_ON USE_AIVLIB_MODEL

CDEFS := $(foreach f, $(ALL_DEFS), $(if $($f),-D$f=$($f)))

# internal flags
NVCCFLAGS   := -Xptxas="-v"   -Xcudafe "--diag_suppress=declared_but_not_referenced"
CCFLAGS     := -O3 -fopenmp -fPIC $(CDEFS)# -DNV=$(NV) -DNA=$(NA) -DNS=$(NS) -DNTIME=$(NTIME) -DDYSH=$(DYSH) -DMPI_ON=$(MPI_ON) 
NVCCLDFLAGS :=
LDFLAGS     := -L/usr/mpi/gcc/openmpi-1.4.2-qlc/lib64/ -L./ -L/usr/lib64/mpich2/lib/

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?= #-std=c++11

ifeq ($(ARCH),k100)
INCLUDES  := -I/usr/mpi/gcc/openmpi-1.4.2-qlc/include/ -I./png/
LIBRARIES := -lmpi -lcudart -lglut -lGL -lcufft -lpng -lgomp -lpthread
else ifeq ($(ARCH),geocluster)
INCLUDES  := -I/usr/mpi/gcc/mvapich-1.2.0/include/
LIBRARIES := -lcudart -lglut -lGL -lcufft -lpng -lgomp -lpthread -lpyaiv2
ifdef MPI_ON
LIBRARIES := -mpich $(LIBRARIES)
endif
else
INCLUDES  := 
LIBRARIES := -lcudart -lglut -lGL -lcufft -lpng -lgomp -lpthread
ifdef USE_AIVLIB_MODEL
LIBRARIES := $(LIBRARIES) -laiv
endif
endif

################################################################################
GENCODE_SM20  := #-gencode arch=compute_20,code=sm_21
GENCODE_SM30  := #-gencode arch=compute_30,code=sm_30
GENCODE_SM35  := #-gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50  := #-arch=sm_20
GENCODE_FLAGS := $(GENCODE_SM50) $(GENCODE_SM35) $(GENCODE_SM30) $(GENCODE_SM20) $(GENCODE_SM)
ALL_CCFLAGS   := --compiler-options="$(CCFLAGS) $(EXTRA_CCFLAGS)" 
ALL_LDFLAGS   := --linker-options="$(LDFLAGS) $(EXTRA_LDFLAGS)"
################################################################################

# Target rules
all: build

DTgeo_wrap.cxx: geo.i params.h py_consts.h texmodel.cuh 
	swig -python -c++ -o DTgeo_wrap.cxx $<
DTgeo_wrap.o: DTgeo_wrap.cxx params.h py_consts.h texmodel.cuh 
	$(GCC) $(INCLUDES) $(CCFLAGS) -c $< -fPIC -I/usr/include/python/ -I/usr/include/python2.7/ -I/usr/include/python2.6/ -I./python2.6/ -o $@
#_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so
ifdef USE_AIVLIB_MODEL
_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so spacemodel/src/space_model.o spacemodel/src/middle_model.o
	$(GCC) $(INCLUDES) $(CCFLAGS) -Wl,-rpath=./ -L./ $(LDFLAGS) $< cudaDTgeo.so -o $@ -shared
else
_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so 
	$(GCC) $(INCLUDES) $(CCFLAGS) -L./ $(LDFLAGS) $< cudaDTgeo.so -o $@ -shared
endif

build: DTgeo _DTgeo.so

kerTFSF.o: kerTFSF.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerTFSF_pmls.o: kerTFSF_pmls.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerITFSF.o: kerITFSF.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerITFSF_pmls.o: kerITFSF_pmls.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

#AsyncTYPES := D_pmls S_pmls I_pmls X_pmls DD_pmls D S I X DD
AsyncTYPES := D_pmls S_pmls I_pmls X_pmls D S I X
obj_files = $(foreach a,$(AsyncTYPES), ker$a.o)

ker%.o: ker%.inc.cu
ker%.o: %.inc.cu
$(obj_files): cuda_math.h params.h py_consts.h defs.h texmodel.cuh 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $(subst .o,.inc.cu,$@)

dt.o: DTgeo.cu diamond.cu im3D.hpp im2D.h cuda_math.h params.h py_consts.h texmodel.cuh init.h signal.hpp window.hpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

TEXMODEL_DEPS := texmodel.cu texmodel.cuh params.h py_consts.h cuda_math.h
ifdef USE_AIVLIB_MODEL
TEXMODEL_DEPS := $(TEXMODEL_DEPS) spacemodel/include/access2model.hpp
obj_files := spacemodel/src/space_model.o spacemodel/src/middle_model.o $(obj_files)
endif

texmodel.o: $(TEXMODEL_DEPS)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
DTgeo: texmodel.o dt.o im3D.o kerTFSF.o kerTFSF_pmls.o kerITFSF.o kerITFSF_pmls.o $(obj_files)
ifndef USE_AIVLIB_MODEL
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LDFLAGS) -o $@ $+ $(LIBRARIES)
endif
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LDFLAGS) -o cudaDTgeo.so $+ $(LIBRARIES) --shared

im3D.o: im3D.cu im3D.hpp cuda_math.h fpal.h im2D.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

cudaDTgeo.so: DTgeo
#DTgeo: texmodel.o dt.o im3D.o kerTFSF.o kerTFSF_pmls.o $(obj_files)

run: build
	$(EXEC) ./DTgeo

clean:
	$(EXEC) rm -f dt.o texmodel.o im3D.o ker*.o DTgeo _DTgeo.so cudaDTgeo.so DTgeo_wrap* DTgeo.py DTgeo.pyc

clobber: clean

