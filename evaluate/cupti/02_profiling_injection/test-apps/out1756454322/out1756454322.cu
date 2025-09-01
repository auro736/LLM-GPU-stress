bash
# Install CUDA toolkit if not already installed
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version

# If nvcc is not found, add CUDA binaries to PATH
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify nvcc is in PATH
which nvcc