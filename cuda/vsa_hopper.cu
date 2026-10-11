// Standalone C ABI for FastVideo's Hopper tile-64 BF16 forward.
// No ATen, libtorch, Python runtime, or training code is linked.
#include "vsa_hopper.h"
#include "vendor/vsa_hopper_forward.cuh"

cudaError_t h3_vsa_hopper(void *out, const void *q, const void *k, const void *v,
                         float *lse, int *indices, int *counts, int *sizes,
                         unsigned tiles, unsigned heads, cudaStream_t stream) {
    int device = 0, major = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return status;
    if (major != 9 || !tiles || !heads) return cudaErrorNotSupported;
    using G = fwd_globals<128>;
    unsigned rows = tiles * 64;
    G g{
        G::q_gl{(bf16 *)q, 1U, heads, rows, 128U},
        G::k_gl{(bf16 *)k, 1U, heads, rows, 128U},
        G::v_gl{(bf16 *)v, 1U, heads, rows, 128U},
        G::l_gl{lse, 1U, heads, 1U, rows},
        G::o_gl{(bf16 *)out, 1U, heads, rows, 128U},
        (int)rows, 1, (int)tiles, indices, counts, sizes
    };
    status = cudaFuncSetAttribute(fwd_attend_ker<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 54000);
    if (status != cudaSuccess) return status;
    fwd_attend_ker<128><<<dim3(tiles, heads, 1), 128, 54000, stream>>>(g);
    return cudaGetLastError();
}
