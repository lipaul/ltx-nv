# CUDA C standalone — build with the nvcc 13.2 shipped inside reference/.venv (cu13 layout).
NVCC := $(CURDIR)/../reference/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
CUDA := $(CURDIR)/../reference/.venv/lib/python3.12/site-packages/nvidia/cu13
CUDNN := $(CURDIR)/../reference/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
ARCH := -arch=sm_89
NVCCFLAGS := $(ARCH) -O2 -std=c++17 --expt-relaxed-constexpr -Icommon \
             -Xlinker -rpath=$(CUDA)/lib -Xlinker -rpath=$(CUDNN) \
             -L$(CUDA)/lib -L$(CUDNN)
LDLIBS := -l:libcudart.so.13 -l:libcublas.so.13 -l:libcublasLt.so.13 -l:libcudnn.so.9

APPS := tests/m1_philox tests/m1_linear dit/test_block runner/run_all

all: $(APPS)

tests/m1_philox: tests/m1_philox.cu common/common.cu common/philox.cu common/st.cpp common/json.cpp common/dump.cpp
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

tests/m1_linear: tests/m1_linear.cu common/common.cu common/gemm.cu common/st.cpp common/json.cpp common/dump.cpp common/philox.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

dit/test_block: dit/test_block.cu common/common.cu common/gemm.cu common/st.cpp common/json.cpp common/dump.cpp common/philox.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

runner/run_all: runner/main.cu dit/model.cu dit/sampler.cu vae/video_decoder.cu vae/audio_decoder.cu \
                vae/vocoder.cu te/feature_extractor.cu te/connector.cu te/gemma.cu \
                common/common.cu common/gemm.cu common/st.cpp common/json.cpp common/dump.cpp common/philox.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(APPS)
