// Minimal read-only JSON parser (safetensors headers + config metadata).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct JVal {
    enum Kind { OBJ, ARR, STR, NUM, BOOL, NUL } kind = NUL;
    std::map<std::string, JVal> obj;
    std::vector<JVal> arr;
    std::string str;
    double num = 0;
    bool b = false;

    bool has(const std::string& k) const { return kind == OBJ && obj.count(k); }
    const JVal& at(const std::string& k) const { return obj.at(k); }
    int64_t as_int() const { return (int64_t)num; }
    std::vector<int64_t> as_int_vec() const {
        std::vector<int64_t> v;
        for (auto& e : arr) v.push_back((int64_t)e.num);
        return v;
    }
    const std::string& as_str() const { return str; }
};

// Parses `text` (must stay valid is NOT required — strings are copied).
JVal json_parse(const char* text, size_t len);
