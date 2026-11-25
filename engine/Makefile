NVCC ?= nvcc
ARCH ?= sm_75
OUT ?= build/libinferio.so
SRC = v0/inferio_v0.cu

all: $(OUT)

$(OUT): $(SRC) inferio.h
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -Xcompiler -fPIC -arch=$(ARCH) -lineinfo -I. -shared $(SRC) -lcublas -o $(OUT)

clean:
	rm -rf build
