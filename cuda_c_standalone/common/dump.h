// Binary tensor dump + JSON manifest, mirroring the standalone/ artifact layout.
#pragma once
#include "tensor.h"
#include <string>
#include <vector>

struct DumpWriter {
    std::string dir;
    // manifest.json lines appended as dumps happen
    std::string manifest;

    void begin(const std::string& dir_);
    void dump(const std::string& name, const Tensor& t);  // writes dir/name + manifest entry
    void end();                                           // writes manifest.json
};
