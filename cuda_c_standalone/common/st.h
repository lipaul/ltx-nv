// safetensors loader: mmap file, parse header, copy tensors to GPU (with optional cast).
#pragma once
#include "json.h"
#include "tensor.h"
#include <unordered_map>
#include <functional>

class SafeTensorsFile {
   public:
    struct Entry {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t begin, end;
    };
    ~SafeTensorsFile();
    void open(const std::string& path);
    const JVal& metadata() const { return meta_; }
    JVal metadata_parsed(const std::string& key) const;  // metadata string field re-parsed as JSON
    bool has(const std::string& key) const { return entries_.count(key) != 0; }
    const Entry& entry(const std::string& key) const { return entries_.at(key); }
    std::vector<std::string> keys() const;

    // Copy tensor `key` to GPU. Floating sources are cast to `float_dtype` (BF16 or F32).
    Tensor load(const std::string& key, DType float_dtype, Arena& arena) const;

    // Load all tensors whose renamed key is non-null. rename: ckpt_key -> state_key | "".
    // `float_dtype`: cast target for floating point tensors.
    std::unordered_map<std::string, Tensor> load_state_dict(
        const std::function<const char*(const std::string&)>& rename, DType float_dtype, Arena& arena) const;

   private:
    int fd_ = -1;
    void* map_ = nullptr;
    size_t size_ = 0;
    size_t data_begin_ = 0;
    JVal meta_;
    std::unordered_map<std::string, Entry> entries_;
};
