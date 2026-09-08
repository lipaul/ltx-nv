#include "dump.h"
#include <sys/stat.h>
#include <cstdio>

static void mkdirs(const std::string& d) { mkdir(d.c_str(), 0755); }

void DumpWriter::begin(const std::string& dir_) {
    dir = dir_;
    mkdirs(dir);
    manifest = "";
}

void DumpWriter::dump(const std::string& name, const Tensor& t) {
    std::string path = dir + "/" + name;
    void* h = to_host(t);
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(h, 1, t.nbytes(), f);
    fclose(f);
    free(h);
    manifest += "\"" + name + "\": {\"dtype\": \"" + dtype_name(t.dtype) + "\", \"shape\": [";
    for (size_t i = 0; i < t.shape.size(); i++) manifest += (i ? ", " : "") + std::to_string(t.shape[i]);
    manifest += "]},\n";
}

void DumpWriter::end() {
    FILE* f = fopen((dir + "/manifest.json").c_str(), "w");
    fputs("{\n", f);
    fputs(manifest.c_str(), f);
    fputs("\"_end\": 0\n}\n", f);
    fclose(f);
}
