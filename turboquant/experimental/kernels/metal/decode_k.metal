    uint gid = thread_position_in_grid.x;
    
    uint d_g = N_GROUPS * GROUP_SIZE;
    uint elements_in_uint = 32u / BITS;
    
    uint prefix_idx = gid / d_g;
    uint local_idx_in_row = gid % d_g;
    
    uint g = local_idx_in_row / GROUP_SIZE;
    uint scale_idx = prefix_idx * N_GROUPS + g;

    // unpack
    uint word_idx_in_row = local_idx_in_row / elements_in_uint;
    uint global_word_idx = prefix_idx * N_WORDS + word_idx_in_row;
    uint word = packed[global_word_idx];

    uint shift = (local_idx_in_row % elements_in_uint) * BITS;
    uint code = (word >> shift) & ((1u << BITS) - 1u);

    uint q_max = (1u << (BITS - 1u)) - 1u;
    int signed_code = (int)code - (int)q_max;

    float scale = (float)scales[scale_idx];
    float val = (float((float)signed_code) * scale);

    // residual add
    uint local_idx = local_idx_in_row % GROUP_SIZE;

    // Only do residual if TOPK > 0
    #pragma unroll
    for (uint i = 0; i < TOPK; i++) {
        // TOPK > 0 so this compiles or doesn't depending on TOPK template
        bool match = (resid_idx[scale_idx * TOPK + i] == local_idx);
        val += match ? (float)resid_vals[scale_idx * TOPK + i] : 0.0f;
    }

    if (local_idx_in_row < D_HEAD) { out[prefix_idx * D_HEAD + local_idx_in_row] = half(val); }
