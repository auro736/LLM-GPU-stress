# Path to CUDA (adjust if not in default location)
CUDAPATH := /usr/local/cuda

# CUDA compiler
NVCC := $(CUDAPATH)/bin/nvcc

# Target executable
TARGET := out1757948074V0

# Source file
SRC := out1757948074.cu

# Compiler flags
NVCCFLAGS := -O3 -std=c++11 -arch=sm_86

# Default rule
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)
