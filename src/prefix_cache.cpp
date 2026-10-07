#include "inferno/prefix_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace inferno {

std::uint64_t PrefixCache::hash_tokens(const std::vector<int>& toks, int len) {
    // FNV-1a 64-bit over the shared prefix.
    std::uint64_t h = 14695981039346656037ull;
    for (int i = 0; i < len; ++i) {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(toks[static_cast<std::size_t>(i)]));
        h *= 1099511628211ull;
    }
    h ^= static_cast<std::uint64_t>(len) * 0x9e3779b97f4a7c15ull;
    return h;
}

int PrefixCache::shared_len(const std::vector<int>& prompt, int requested_prefix_len) const {
    if (block_size_ < 1 || prompt.empty()) return 0;
    int L = requested_prefix_len;
    if (L <= 0) L = static_cast<int>(prompt.size());
    L = std::min(L, static_cast<int>(prompt.size()));
    L = (L / block_size_) * block_size_;
    return L;
}

PrefixEntry* PrefixCache::find(std::uint64_t h) {
    auto it = map_.find(h);
    if (it == map_.end()) return nullptr;
    return &it->second;
}

PrefixEntry& PrefixCache::insert(std::uint64_t h, std::vector<int> tokens, std::vector<int> blocks,
                                 int owner) {
    if (map_.count(h)) throw std::runtime_error("prefix entry already exists");
    PrefixEntry e;
    e.hash = h;
    e.tokens = std::move(tokens);
    e.blocks = std::move(blocks);
    e.refcount = 1;
    e.ready = false;
    e.owner_slot = owner;
    auto [it, ok] = map_.emplace(h, std::move(e));
    (void)ok;
    return it->second;
}

void PrefixCache::add_ref(std::uint64_t h) {
    auto* e = find(h);
    if (!e) throw std::runtime_error("prefix add_ref miss");
    ++e->refcount;
}

std::vector<int> PrefixCache::release(std::uint64_t h, bool retain) {
    auto it = map_.find(h);
    if (it == map_.end()) return {};
    PrefixEntry& e = it->second;
    if (e.refcount < 1) throw std::runtime_error("prefix refcount underflow");
    --e.refcount;
    if (e.refcount > 0) return {};
    std::vector<int> blocks;
    if (!retain) {
        blocks = e.blocks;
        map_.erase(it);
    } else {
        // Keep pages warm for the next batch; refcount stays 0 until re-attached.
        e.owner_slot = -1;
    }
    return blocks;
}

void PrefixCache::mark_ready(std::uint64_t h) {
    auto* e = find(h);
    if (!e) return;
    e->ready = true;
}

void PrefixCache::clear() { map_.clear(); }

}  // namespace inferno
