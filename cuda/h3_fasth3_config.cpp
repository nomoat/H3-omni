#include "../h3_fasth3.h"
#include "vendor/json.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

extern "C" int h3_fasth3_config_read(const char *directory, h3_fasth3_config *out,
                                     char *error, size_t capacity) {
    if (!directory || !out) return 0;
    *out = {};
    std::string path = std::string(directory) + "/h3_fasth3.json";
    struct stat st{};
    if (stat(path.c_str(), &st)) {
        if (errno == ENOENT) return 1;
        if (error && capacity) snprintf(error, capacity, "cannot stat %s: %s", path.c_str(), strerror(errno));
        return 0;
    }
    try {
        if (st.st_size <= 0 || st.st_size > 65536) throw std::runtime_error("invalid manifest size");
        std::ifstream file(path);
        nlohmann::json j = nlohmann::json::parse(file);
        if (j.at("format") != "h3.c-fasth3-v1") throw std::runtime_error("unsupported export format");
        auto kind = j.at("attention").get<std::string>();
        if (kind != "dense" && kind != "vsa") throw std::runtime_error("unsupported attention backend");
        if (j.at("task") != "t2av" || j.at("guidance_scale") != 1.0)
            throw std::runtime_error("only unguided T2VA is supported");
        auto rungs = j.at("dmd_denoising_steps");
        const int expected[] = {999, 749, 500, 250};
        if (!rungs.is_array() || rungs.size() != 4) throw std::runtime_error("expected four DMD rungs");
        for (int i = 0; i < 4; ++i)
            if (!rungs[i].is_number_integer() || rungs[i].get<int>() != expected[i])
                throw std::runtime_error("unsupported DMD schedule");
        if (j.at("video_shift") != 12.0 || j.at("audio_shift") != 3.0)
            throw std::runtime_error("unsupported scheduler shifts");
        if (kind == "vsa" && (j.at("tile_size") != 64 || j.at("sparsity") != 0.9))
            throw std::runtime_error("VSA requires tile size 64 and sparsity 0.9");
        out->kind = kind == "vsa" ? H3_FASTH3_VSA : H3_FASTH3_DENSE;
        out->steps = 4;
        std::copy(expected, expected + 4, out->rungs);
        out->video_shift = 12.0f;
        out->audio_shift = 3.0f;
        return 1;
    } catch (const std::exception &e) {
        if (error && capacity) snprintf(error, capacity, "invalid FastH3 manifest: %s", e.what());
        *out = {};
        return 0;
    }
}

extern "C" int h3_live_config_read(const char *directory, int *enabled,
                                   char *error, size_t capacity) {
    *enabled = 0;
    std::string path = std::string(directory) + "/h3_live.json";
    struct stat st{};
    if (stat(path.c_str(), &st)) {
        if (errno == ENOENT) return 1;
        if (error && capacity) snprintf(error, capacity, "cannot read live manifest: %s", strerror(errno));
        return 0;
    }
    try {
        if (st.st_size <= 0 || st.st_size > 65536) throw std::runtime_error("invalid manifest size");
        std::ifstream file(path);
        auto j = nlohmann::json::parse(file);
        if (j.at("format") != "h3.c-live-v1" || j.at("model") != "TaoMate-H3" ||
            j.at("rank") != 128 || j.at("alpha") != 128 ||
            j.at("qkv_layout") != "head-qkv-dim" ||
            j.at("state_indices") != nlohmann::json({0,16,33,49}) ||
            j.at("video_shift") != 12 || j.at("audio_shift") != 3)
            throw std::runtime_error("unsupported TaoMate-H3 contract");
        *enabled = 1;
        return 1;
    } catch (const std::exception &e) {
        if (error && capacity) snprintf(error, capacity, "invalid live manifest: %s", e.what());
        return 0;
    }
}
