#pragma once

#include <vector>

namespace inferno {

// Paged KV cache.
// Layout per layer: [num_blocks, n_kv_heads, block_size, head_dim]
// A sequence's logical token at `pos` lives at
//   physical = block_table[pos / block_size], slot = pos % block_size.
class PagedCache {
public:
    PagedCache(int n_layers, int n_kv_heads, int head_dim, int num_blocks, int block_size);

    int alloc_block();
    void free_block(int block);
    void free_all(const std::vector<int>& blocks);

    int num_blocks() const { return num_blocks_; }
    int block_size() const { return block_size_; }
    int free_blocks() const { return static_cast<int>(free_.size()); }
    int n_layers() const { return n_layers_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }

    void set_block_table(int seq, const std::vector<int>& table);
    void clear_seq(int seq);
    const std::vector<int>& block_table(int seq) const;

    void write_kv(int layer, int seq, int pos, int kv_head,
                  const float* k, const float* v);
    const float* k_ptr(int layer, int block, int kv_head, int slot) const;
    const float* v_ptr(int layer, int block, int kv_head, int slot) const;

    // FlashAttention-style online softmax walked one page at a time.
    // q/out: [T, n_heads, head_dim]
    void attention(const float* q, const int* seq_ids, const int* positions, int tokens,
                   int layer, int n_heads, int n_kv_heads, int head_dim, float* out) const;

private:
    int index(int block, int kv_head, int slot) const;
    int n_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int num_blocks_ = 0;
    int block_size_ = 0;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<int> free_;
    std::vector<std::vector<int>> tables_;
};

}  // namespace inferno
