#include "st.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>

SafeTensorsFile::~SafeTensorsFile() {
    if (map_) munmap(map_, size_);
    if (fd_ >= 0) close(fd_);
}

void SafeTensorsFile::open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    struct stat st;
    fstat(fd_, &st);
    size_ = st.st_size;
    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path.c_str());
        exit(1);
    }
    uint64_t hdr_len;
    memcpy(&hdr_len, map_, 8);
    const char* hdr = (const char*)map_ + 8;
    JVal root = json_parse(hdr, hdr_len);
    data_begin_ = 8 + hdr_len;
    for (auto& [key, val] : root.obj) {
        if (key == "__metadata__") {
            meta_ = val;
            continue;
        }
        Entry e;
        e.dtype = val.at("dtype").as_str();
        e.shape = val.at("shape").as_int_vec();
        auto off = val.at("data_offsets").as_int_vec();
        e.begin = data_begin_ + off[0];
        e.end = data_begin_ + off[1];
        entries_[key] = e;
    }
}

JVal SafeTensorsFile::metadata_parsed(const std::string& key) const {
    const std::string& s = meta_.at(key).as_str();
    return json_parse(s.data(), s.size());
}

std::vector<std::string> SafeTensorsFile::keys() const {
    std::vector<std::string> ks;
    for (auto& [k, _] : entries_) ks.push_back(k);
    std::sort(ks.begin(), ks.end());
    return ks;
}

static DType map_dtype(const std::string& s) {
    if (s == "BF16") return BF16;
    if (s == "F32") return F32;
    if (s == "F64") return F64;
    if (s == "I64") return I64;
    if (s == "U8") return U8;
    fprintf(stderr, "unsupported safetensors dtype %s\n", s.c_str());
    exit(1);
}

Tensor SafeTensorsFile::load(const std::string& key, DType float_dtype, Arena& arena) const {
    const Entry& e = entries_.at(key);
    DType src = map_dtype(e.dtype);
    Tensor t;
    t.shape = e.shape;
    t.dtype = src;
    if (src == F32 || src == F64) t.dtype = float_dtype;
    int64_t n = t.numel();
    const void* host = (const char*)map_ + e.begin;

    if (src == t.dtype) {
        t.data = arena.alloc(t.nbytes());
        CUDA_CHECK(cudaMemcpy(t.data, host, t.nbytes(), cudaMemcpyHostToDevice));
        return t;
    }
    // cast path: copy raw to temp device buffer, cast into arena
    void* tmp;
    int64_t src_bytes = n * itemsize(src);
    CUDA_CHECK(cudaMalloc(&tmp, src_bytes));
    CUDA_CHECK(cudaMemcpy(tmp, host, src_bytes, cudaMemcpyHostToDevice));
    t.data = arena.alloc(n * itemsize(t.dtype));
    if (src == F32 && t.dtype == BF16) cast_fp32_to_bf16((const float*)tmp, (bf16*)t.data, n);
    else if (src == BF16 && t.dtype == F32) cast_bf16_to_fp32((const bf16*)tmp, (float*)t.data, n);
    else {
        fprintf(stderr, "no cast path %s -> %s\n", dtype_name(src), dtype_name(t.dtype));
        exit(1);
    }
    CUDA_CHECK(cudaFree(tmp));
    return t;
}

std::unordered_map<std::string, Tensor> SafeTensorsFile::load_state_dict(
    const std::function<const char*(const std::string&)>& rename, DType float_dtype, Arena& arena) const {
    std::unordered_map<std::string, Tensor> sd;
    for (auto& [key, _] : entries_) {
        const char* nk = rename(key);
        if (!nk || !*nk) continue;
        sd[nk] = load(key, float_dtype, arena);
    }
    return sd;
}

// ---------------------------------------------------------------------------
// elementwise kernels live in common.cu (CUDA syntax); host wrappers here.
// ---------------------------------------------------------------------------

void* cuda_malloc_copy(const void* host, int64_t bytes) {
    void* d;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice));
    return d;
}

void* to_host(const Tensor& t) {
    void* h = malloc(t.nbytes());
    CUDA_CHECK(cudaMemcpy(h, t.data, t.nbytes(), cudaMemcpyDeviceToHost));
    return h;
}
