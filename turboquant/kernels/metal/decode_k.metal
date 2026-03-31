#include <metal_stdlib>
using namespace metal;

kernel void decode_k(
    device const uint *packed [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const ushort *resid_idx [[buffer(2)]],
    device const half *resid_vals [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant int &bits [[buffer(5)]],
    constant int &group_size [[buffer(6)]],
    constant int &topk [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]
) {
    int g = gid / group_size;

    // unpack
    int elements_in_uint = 32 / bits;
    uint word = packed[gid / elements_in_uint];
    int shift = (gid % elements_in_uint) * bits;
    uint code = (word >> shift) & ((1 << bits) - 1);

    float scale = (float)scales[g];
    float val = (float(code) * scale);

    // residual add using simdgroup operations if possible, otherwise threadgroup
    int local_idx = gid % group_size;
    for (int i = 0; i < topk; i++) {
        if (resid_idx[g * topk + i] == local_idx) {
            val += (float)resid_vals[g * topk + i];
        }
    }

    out[gid] = half(val);
}
