#include "inferno/paged_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace inferno {

PagedCache::PagedCache(int n_layers, int n_kv_heads, int head_dim, int num_blocks, int block_size)
    : n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      num_blocks_(num_blocks),
      block_size_(block_size) {
    if (n_layers < 1 || n_kv_heads < 1 || head_dim < 1 || num_blocks < 1 || block_size < 1) {
        throw std::runtime_error("invalid paged cache shape");
    }
    const std::size_t per_layer = static_cast<std::size_t>(num_blocks) * n_kv_heads * block_size * head_dim;
    k_.assign(static_cast<std::size_t>(n_layers) * per_layer, 0.f);
    v_.assign(static_cast<std::size_t>(n_layers) * per_layer, 0.f);
    free_.reserve(num_blocks);
    for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);
}

int PagedCache::index(int block, int kv_head, int slot) const {
    return ((block * n_kv_heads_ + kv_head) * block_size_ + slot) * head_dim_;
}

int PagedCache::alloc_block() {
    if (free_.empty()) return -1;
    int b = free_.back();
    free_.pop_back();
    return b;
}

void PagedCache::free_block(int block) {
    if (block < 0 || block >= num_blocks_) throw std::runtime_error("free bad block");
    free_.push_back(block);
}

void PagedCache::free_all(const std::vector<int>& blocks) {
    for (int b : blocks) free_block(b);
}

void PagedCache::set_block_table(int seq, const std::vector<int>& table) {
    if (seq < 0) throw std::runtime_error("bad seq id");
    if (static_cast<int>(tables_.size()) <= seq) tables_.resize(seq + 1);
    tables_[seq] = table;
}

void PagedCache::clear_seq(int seq) {
    if (seq >= 0 && seq < static_cast<int>(tables_.size())) tables_[seq].clear();
}

const std::vector<int>& PagedCache::block_table(int seq) const {
    if (seq < 0 || seq >= static_cast<int>(tables_.size())) {
        throw std::runtime_error("missing block table");
    }
    return tables_[seq];
}

const float* PagedCache::k_ptr(int layer, int block, int kv_head, int slot) const {
    const std::size_t per = static_cast<std::size_t>(num_blocks_) * n_kv_heads_ * block_size_ * head_dim_;
    return k_.data() + layer * per + index(block, kv_head, slot);
}

const float* PagedCache::v_ptr(int layer, int block, int kv_head, int slot) const {
    const std::size_t per = static_cast<std::size_t>(num_blocks_) * n_kv_heads_ * block_size_ * head_dim_;
    return v_.data() + layer * per + index(block, kv_head, slot);
}

void PagedCache::write_kv(int layer, int seq, int pos, int kv_head, const float* k, const float* v) {
    const auto& table = block_table(seq);
    int logical = pos / block_size_;
    int slot = pos % block_size_;
    if (logical < 0 || logical >= static_cast<int>(table.size())) {
        throw std::runtime_error("kv write past block table");
    }
    int block = table[logical];
    const std::size_t per = static_cast<std::size_t>(num_blocks_) * n_kv_heads_ * block_size_ * head_dim_;
    float* kd = k_.data() + layer * per + index(block, kv_head, slot);
    float* vd = v_.data() + layer * per + index(block, kv_head, slot);
    std::memcpy(kd, k, sizeof(float) * head_dim_);
    std::memcpy(vd, v, sizeof(float) * head_dim_);
}

void PagedCache::attention(const float* q, const int* seq_ids, const int* positions, int tokens,
                           int layer, int n_heads, int n_kv_heads, int head_dim, float* out) const {
    if (n_kv_heads != n_kv_heads_ || head_dim != head_dim_) {
        throw std::runtime_error("attention shape does not match cache");
    }
    const int group = n_heads / n_kv_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t per = static_cast<std::size_t>(num_blocks_) * n_kv_heads_ * block_size_ * head_dim_;
    const float* kbase = k_.data() + static_cast<std::size_t>(layer) * per;
    const float* vbase = v_.data() + static_cast<std::size_t>(layer) * per;

    for (int t = 0; t < tokens; ++t) {
        const auto& table = block_table(seq_ids[t]);
        const int pos = positions[t];
        for (int h = 0; h < n_heads; ++h) {
            const int kvh = h / group;
            const float* qq = q + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            float* oo = out + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            float m_i = -INFINITY;
            float l_i = 0.f;
            for (int d = 0; d < head_dim; ++d) oo[d] = 0.f;
            const int kv_len = pos + 1;
            for (int s = 0; s < kv_len; ++s) {
                int logical = s / block_size_;
                int slot = s % block_size_;
                if (logical >= static_cast<int>(table.size())) {
                    throw std::runtime_error("attention past block table");
                }
                int block = table[logical];
                const float* kk = kbase + index(block, kvh, slot);
                const float* vv = vbase + index(block, kvh, slot);
                float score = 0.f;
                for (int d = 0; d < head_dim; ++d) score += qq[d] * kk[d];
                score *= scale;
                float m_new = std::max(m_i, score);
                float p = std::exp(score - m_new);
                float alpha = std::exp(m_i - m_new);
                for (int d = 0; d < head_dim; ++d) oo[d] = oo[d] * alpha + p * vv[d];
                l_i = l_i * alpha + p;
                m_i = m_new;
            }
            float inv = l_i == 0.f ? 0.f : 1.f / l_i;
            for (int d = 0; d < head_dim; ++d) oo[d] *= inv;
        }
    }
}

}  // namespace inferno
