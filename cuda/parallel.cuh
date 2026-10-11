// Multi-GPU support for one process: a device selector, NCCL communicators
// (one per device, ncclCommInitAll) and the two exchanges of Ulysses sequence
// parallelism. Included at the end of h3_gpu_cuda.cu.
//
// Ulysses layout. Each rank holds `rows` consecutive sequence rows of a
// [rows][heads][dim] BF16 tensor. scatter_heads turns it into
// [ranks * rows][heads / ranks][dim]: all sequence rows, this rank's heads.
// gather_heads is the inverse. Both use a scratch tensor of the same size.
// The exchange moves bytes only, so values are unchanged.
#include <nccl.h>

struct h3_gpu_comm {
    ncclComm_t comm;
    int rank;
    int size;
};

static int nccl_ok(h3_gpu *g, ncclResult_t r, const char *what) {
    if (r == ncclSuccess) return 1;
    h3_gpu_set_error(g, "NCCL %s: %s", what, ncclGetErrorString(r));
    return 0;
}

extern "C" int h3_gpu_device_count(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

extern "C" int h3_gpu_select_device(int device, char *error, size_t size) {
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) return 1;
    if (error && size)
        snprintf(error, size, "cannot select CUDA device %d: %s", device, cudaGetErrorString(e));
    return 0;
}

extern "C" int h3_gpu_comm_init_all(int count, h3_gpu_comm **comms, char *error, size_t size) {
    if (count < 1 || count > 64 || !comms) {
        if (error && size) snprintf(error, size, "invalid NCCL communicator count %d", count);
        return 0;
    }
    std::vector<ncclComm_t> raw(count);
    std::vector<int> devices(count);
    for (int i = 0; i < count; i++) devices[i] = i;
    ncclResult_t r = ncclCommInitAll(raw.data(), count, devices.data());
    if (r != ncclSuccess) {
        if (error && size) snprintf(error, size, "NCCL init: %s", ncclGetErrorString(r));
        return 0;
    }
    for (int i = 0; i < count; i++) comms[i] = new h3_gpu_comm{raw[i], i, count};
    return 1;
}

extern "C" void h3_gpu_comm_free(h3_gpu_comm *c) {
    if (!c) return;
    if (c->comm) ncclCommDestroy(c->comm);
    delete c;
}

// Unblocks every rank waiting in a collective after one rank failed. The
// communicator is unusable afterwards.
extern "C" void h3_gpu_comm_abort(h3_gpu_comm *c) {
    if (c) ncclCommAbort(c->comm);
    if (c) c->comm = nullptr;
}

// out[b][a][c] = in[a][b][c]; c in 16-byte units.
__global__ void swap_leading(uint4 *out, const uint4 *in, uint a_count, uint b_count,
                             uint c_count) {
    size_t total = size_t(a_count) * b_count * c_count;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += size_t(gridDim.x) * blockDim.x) {
        uint c = uint(i % c_count);
        size_t ab = i / c_count;
        uint b = uint(ab % b_count), a = uint(ab / b_count);
        out[(size_t(b) * a_count + a) * c_count + c] = in[i];
    }
}

static int swap_leading_bf16(h3_gpu *g, h3_gpu_tensor *out, const h3_gpu_tensor *in, uint a,
                             uint b, size_t c_elements) {
    size_t bytes = c_elements * 2;
    if (bytes % 16) {
        h3_gpu_set_error(g, "Ulysses block of %zu bytes is not 16-byte aligned", bytes);
        return 0;
    }
    uint c = uint(bytes / 16);
    size_t total = size_t(a) * b * c;
    uint blocks = uint(std::min<size_t>((total + 255) / 256, 65535));
    swap_leading<<<blocks, 256, 0, g->stream>>>((uint4 *)out->data, (const uint4 *)in->data, a,
                                                b, c);
    return launch_ok(g);
}

// Every rank sends block j of `send` to rank j and receives rank j's block
// into block j of `recv`.
static int all_to_all(h3_gpu *g, h3_gpu_comm *c, void *recv, const void *send, size_t block_bytes) {
    if (!nccl_ok(g, ncclGroupStart(), "group start")) return 0;
    for (int peer = 0; peer < c->size; peer++) {
        ncclResult_t r = ncclSend((const char *)send + peer * block_bytes, block_bytes, ncclUint8,
                                  peer, c->comm, g->stream);
        if (r == ncclSuccess)
            r = ncclRecv((char *)recv + peer * block_bytes, block_bytes, ncclUint8, peer, c->comm,
                         g->stream);
        if (r != ncclSuccess) {
            ncclGroupEnd();
            return nccl_ok(g, r, "all-to-all");
        }
    }
    return nccl_ok(g, ncclGroupEnd(), "group end");
}

static int ulysses_check(h3_gpu *g, h3_gpu_comm *c, const h3_gpu_tensor *t,
                         const h3_gpu_tensor *scratch, uint rows, uint heads, uint dim) {
    size_t count = size_t(rows) * heads * dim;
    if (!c || !c->comm || heads % c->size) {
        h3_gpu_set_error(g, "Ulysses: %u heads do not split over %d ranks", heads,
                         c ? c->size : 0);
        return 0;
    }
    return h3_gpu_require_bf16(g, t, count, "Ulysses tensor") &&
           h3_gpu_require_bf16(g, scratch, count, "Ulysses scratch");
}

extern "C" int h3_gpu_ulysses_scatter_heads_bf16(h3_gpu *g, h3_gpu_comm *c, h3_gpu_tensor *t,
                                                 h3_gpu_tensor *scratch, uint rows, uint heads,
                                                 uint dim) {
    if (!ulysses_check(g, c, t, scratch, rows, heads, dim)) return 0;
    size_t group = size_t(heads / c->size) * dim;
    // [rows][ranks][group] -> [ranks][rows][group], then block j to rank j.
    return swap_leading_bf16(g, scratch, t, rows, c->size, group) &&
           all_to_all(g, c, t->data, scratch->data, size_t(rows) * group * 2);
}

extern "C" int h3_gpu_ulysses_gather_heads_bf16(h3_gpu *g, h3_gpu_comm *c, h3_gpu_tensor *t,
                                                h3_gpu_tensor *scratch, uint rows, uint heads,
                                                uint dim) {
    if (!ulysses_check(g, c, t, scratch, rows, heads, dim)) return 0;
    size_t group = size_t(heads / c->size) * dim;
    // Rows block j back to rank j: [ranks][rows][group] -> [rows][ranks][group].
    return all_to_all(g, c, scratch->data, t->data, size_t(rows) * group * 2) &&
           swap_leading_bf16(g, t, scratch, c->size, rows, group);
}

// output[rank * count + i] = input_of_rank[i] for every rank.
extern "C" int h3_gpu_all_gather_bf16(h3_gpu *g, h3_gpu_comm *c, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *input, size_t count) {
    if (!c || !c->comm || !h3_gpu_require_bf16(g, input, count, "all-gather input") ||
        !h3_gpu_require_bf16(g, output, count * c->size, "all-gather output"))
        return 0;
    return nccl_ok(g, ncclAllGather(input->data, output->data, count * 2, ncclUint8, c->comm,
                                    g->stream), "all-gather");
}

extern "C" int h3_gpu_zero(h3_gpu *g, h3_gpu_tensor *t) {
    return t && t->owner == g && cuda_ok(g, cudaMemsetAsync(t->data, 0, t->bytes, g->stream));
}
