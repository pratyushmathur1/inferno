#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace inferno {

// Block-aligned automatic prefix KV reuse.
// Sequences that share an identical token prefix of length L (L % block_size == 0)
// share the same physical pages for [0, L). A cold miss computes the prefix once;
// later hits skip that prefill. Concurrent waiters block until the owner finishes.
struct PrefixEntry {
    std::uint64_t hash = 0;
    std::vector<int> tokens;
    std::vector<int> blocks;
    int refcount = 0;
    bool ready = false;
    int owner_slot = -1;
};

class PrefixCache {
public:
    explicit PrefixCache(int block_size) : block_size_(block_size) {}

    static std::uint64_t hash_tokens(const std::vector<int>& toks, int len);

    // Longest block-aligned prefix length to share (0 if none / disabled).
    int shared_len(const std::vector<int>& prompt, int requested_prefix_len) const;

    PrefixEntry* find(std::uint64_t h);
    PrefixEntry& insert(std::uint64_t h, std::vector<int> tokens, std::vector<int> blocks, int owner);

    void add_ref(std::uint64_t h);
    // Returns blocks to free when the last reference drops (empty if retained or still live).
    std::vector<int> release(std::uint64_t h, bool retain);

    void mark_ready(std::uint64_t h);
    void clear();

    int size() const { return static_cast<int>(map_.size()); }

private:
    int block_size_ = 16;
    std::unordered_map<std::uint64_t, PrefixEntry> map_;
};

}  // namespace inferno
