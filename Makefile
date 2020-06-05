CXX = g++
CXXFLAGS = -Wall -Werror -Wextra -pedantic -std=c++17 -g #-fsanitize=address
LDFLAGS = -O3 #-fsanitize=address

SRC = test.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = test

NVCC = /usr/local/cuda/bin/nvcc

NVCCFLAGS = \
	--gpu-architecture=compute_61 --gpu-code=compute_61 \
	--compiler-options "-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC"

	# -Xcompiler -fPIC
	# --compiler-options "-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC"

# CUDA_SRC = cudaTest.cu
# CUDA_OBJ = $(CUDA_SRC:.cu=.o)
# CUDA_LIB = libtest.so

CUDA_LIB_NAME = shabal
CUDA_SRC = shabal.cu
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
CUDA_LIB = lib$(CUDA_LIB_NAME).so

all: $(EXEC)

lib: $(CUDA_LIB)

$(EXEC): $(OBJ) $(CUDA_LIB)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(LBLIBS) -L. -l$(CUDA_LIB_NAME)

$(CUDA_LIB): $(CUDA_OBJ)
	$(CXX) -shared -o $@ $^ -L/usr/local/cuda/lib64 -lcudart

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $^ -o $(CUDA_OBJ)

clean:
	rm -rf $(OBJ) $(EXEC) $(CUDA_LIB) $(CUDA_OBJ)