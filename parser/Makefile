DS_ROOT ?= /opt/nvidia/deepstream/deepstream
CUDA_HOME ?= /usr/local/cuda

TARGET := libnvds_e2e_parser.so
SRCS := nvdsparsebbox_e2e.cpp

CXXFLAGS := -O2 -Wall -std=c++17 -shared -fPIC \
	-I$(DS_ROOT)/sources/includes -I$(CUDA_HOME)/include

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: clean
