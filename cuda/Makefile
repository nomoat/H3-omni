ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD ?= $(ROOT)/build-cuda
NVCC ?= nvcc
CC ?= cc
CUDA_ARCH ?= sm_90
# PyTorch's CUDA image installs cuDNN as an NVIDIA Python wheel.
CUDNN_ROOT ?= /opt/conda/lib/python3.11/site-packages/nvidia/cudnn
ICU_ROOT ?= /usr
FLASH_ROOT ?= $(ROOT)/.deps/flash-cuda
CUTLASS_ROOT ?= $(ROOT)/.deps/cutlass-flash
FLASH3_ROOT ?= $(ROOT)/.deps/flash3-hopper
CUTLASS3_ROOT ?= $(ROOT)/.deps/cutlass-flash3
CFLAGS := -O2 -g -std=c11 -ffp-contract=off -D_GNU_SOURCE -DH3_CUDA -I$(ROOT) -fPIC -MMD -MP
NVFLAGS := -O2 -lineinfo -std=c++17 -arch=$(CUDA_ARCH) --fmad=false -Xcompiler=-fPIC -I$(ROOT) -I$(ROOT)/cuda -I$(CUDNN_ROOT)/include
LIB_C := h3 h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_video_encoder h3_audio_vae h3_ffmpeg h3_terminal h3_vision_encoder h3_multimodal
GPU_OBJ := $(BUILD)/h3_gpu_cuda.o $(BUILD)/flash_attention.o $(BUILD)/flash3_attention.o
OBJ := $(addprefix $(BUILD)/,$(addsuffix .o,$(LIB_C))) $(GPU_OBJ)
LDLIBS := -lcublas -lcublasLt -L$(CUDNN_ROOT)/lib -Xlinker=-rpath=$(CUDNN_ROOT)/lib -l:libcudnn.so.9 -lpthread -lm
.PHONY: all primitives clean
all: $(BUILD)/h3 $(BUILD)/libh3_cuda.so $(BUILD)/libh3_model.so $(BUILD)/libh3_tokenizer.so
primitives: $(BUILD)/libh3_cuda.so
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: $(ROOT)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/h3_gpu_cuda.o: $(ROOT)/cuda/h3_gpu_cuda.cu $(ROOT)/cuda/h3_cuda.h $(ROOT)/cuda/h3_flash.h $(ROOT)/h3_gpu.h $(ROOT)/cuda/primitives.cuh $(ROOT)/cuda/fast_primitives.cuh $(ROOT)/cuda/primitive_wrappers.inc $(ROOT)/cuda/dispatch.inc $(ROOT)/cuda/unsupported.inc $(ROOT)/cuda/vae_fused.cuh $(ROOT)/cuda/vae_attention.cuh $(ROOT)/cuda/vae_fp16.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@
$(BUILD)/flash_attention.o: $(ROOT)/cuda/flash_attention.cu $(ROOT)/cuda/h3_flash.h $(FLASH_ROOT)/flash_fwd_kernel.h | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr --expt-extended-lambda -DUNFUSE_FMA -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP -Xcompiler=-fPIC -I$(ROOT)/cuda -I$(FLASH_ROOT) -I$(CUTLASS_ROOT)/include -c $< -o $@
# FlashAttention-3 needs sm_90a (WGMMA/TMA) and its own CUTLASS revision. All
# symbols except the entry point are made local so the two CUTLASS revisions
# cannot resolve to each other's inline functions.
FLASH3_DISABLE := BACKWARD SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA FP8 VARLEN SM8x HDIM96 HDIM192 HDIM256
$(BUILD)/flash3_attention.o: $(ROOT)/cuda/flash3_attention.cu $(ROOT)/cuda/h3_flash.h $(FLASH3_ROOT)/flash_fwd_launch_template.h | $(BUILD)
	$(NVCC) -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG $(addprefix -DFLASHATTENTION_DISABLE_,$(FLASH3_DISABLE)) -Xcompiler=-fPIC -I$(ROOT)/cuda -I$(FLASH3_ROOT) -I$(CUTLASS3_ROOT)/include -c $< -o $(BUILD)/flash3_attention.raw.o
	objcopy --keep-global-symbol=h3_flash3_bf16 --keep-global-symbol=h3_flash3_f16_d64 $(BUILD)/flash3_attention.raw.o $@
$(BUILD)/libh3_cuda.so: $(GPU_OBJ)
	$(NVCC) -shared -o $@ $^ $(LDLIBS)
MODEL_C := h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_audio_vae
$(BUILD)/model_bridge.o: $(ROOT)/tests/pytorch/model_bridge.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/libh3_model.so: $(addprefix $(BUILD)/,$(addsuffix .o,$(MODEL_C))) $(GPU_OBJ) $(BUILD)/model_bridge.o
	$(NVCC) -shared -o $@ $^ $(LDLIBS)
$(BUILD)/h3: $(OBJ) $(BUILD)/main.o $(BUILD)/h3_cli.o $(BUILD)/linenoise.o $(BUILD)/h3_tokenizer_portable.o
	$(NVCC) -o $@ $^ $(LDLIBS) -L$(ICU_ROOT)/lib/x86_64-linux-gnu -Xlinker=--disable-new-dtags -Xlinker=-rpath=$(ICU_ROOT)/lib/x86_64-linux-gnu -licuuc -licui18n
$(BUILD)/serve_check.o: $(ROOT)/tests/serving/serve_check.c $(ROOT)/h3.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/h3_serve_check: $(OBJ) $(BUILD)/serve_check.o $(BUILD)/h3_tokenizer_portable.o
	$(NVCC) -o $@ $^ $(LDLIBS) -L$(ICU_ROOT)/lib/x86_64-linux-gnu -Xlinker=--disable-new-dtags -Xlinker=-rpath=$(ICU_ROOT)/lib/x86_64-linux-gnu -licuuc -licui18n
$(BUILD)/h3_tokenizer_portable.o:$(ROOT)/cuda/h3_tokenizer_portable.cpp $(ROOT)/cuda/vendor/json.hpp $(ROOT)/h3_tokenizer.h | $(BUILD)
	$(CXX) -O2 -std=c++17 -fPIC -I$(ROOT) -I$(ICU_ROOT)/include -c $< -o $@
$(BUILD)/libh3_tokenizer.so: $(BUILD)/h3_tokenizer_portable.o
	$(CXX) -shared -o $@ $^ -L$(ICU_ROOT)/lib/x86_64-linux-gnu -Wl,--disable-new-dtags,-rpath,$(ICU_ROOT)/lib/x86_64-linux-gnu -licuuc -licui18n
clean:
	rm -f $(BUILD)/*.o $(BUILD)/h3 $(BUILD)/libh3_cuda.so

-include $(wildcard $(BUILD)/*.d)
