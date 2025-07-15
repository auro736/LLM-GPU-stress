CUDAPATH = /usr/local/cuda
NVCC =  ${CUDAPATH}/bin/nvcc

# Nome del file eseguibile
TARGET = stress_test

# File sorgente
SRC = out_1752585331.cu

# Opzioni di compilazione
NVCCFLAGS = -O3 -std=c++11

# Regola di default
all: $(TARGET)

# Regola per costruire l'eseguibile
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Pulizia dei file generati
clean:
	rm -f $(TARGET)
