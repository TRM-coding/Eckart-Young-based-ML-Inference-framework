#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef RESNET50_USE_ONEDNN
#include "oneapi/dnnl/dnnl.hpp"
#endif

namespace {

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct image_u8 {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;
};

struct preproc_config {
    uint32_t image_size = 224;
    uint32_t resize_shorter_edge = 256;
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};
};

struct resnet50_model {
    gguf_ptr meta{nullptr, gguf_free};
    ggml_ctx_ptr weights{nullptr, ggml_free};
    preproc_config preproc;
    std::array<uint32_t, 4> stage_block_count = {3, 4, 6, 3};
    std::vector<std::string> labels;
};

struct args {
    std::string model_path;
    std::string image_path;
    std::string mode = "fold";
    int threads = 8;
    int top_k = 5;
    int warmup = 1;
    int repeat = 1;
    bool benchmark = false;
};

struct feature_map {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;
#ifdef RESNET50_USE_ONEDNN
    std::vector<uint8_t> onednn_buffer;
    dnnl::memory onednn_memory;
    bool data_valid = true;
    bool onednn_valid = false;
#endif

    float & at(int c, int y, int x) {
        return data[static_cast<size_t>(x + width * (y + height * c))];
    }

    const float & at(int c, int y, int x) const {
        return data[static_cast<size_t>(x + width * (y + height * c))];
    }
};

struct svd_factors {
    int64_t input_len = 0;
    int64_t rank = 0;
    int64_t output_len = 0;
    std::vector<float> v;
    std::vector<float> u;
    bool from_precomputed_svd = false;
};

struct rank_cols_result {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> cols;
};

#ifdef RESNET50_USE_ONEDNN
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

std::string blob_key(const dnnl::memory::desc & md) {
    auto tmp = md;
    const std::vector<uint8_t> blob = tmp.get_blob();
    return std::string(reinterpret_cast<const char *>(blob.data()), blob.size());
}

struct onednn_unary_key {
    std::string src_blob;
    int op = 0;
    int kernel = 0;
    int stride = 0;
    int padding = 0;

    bool operator==(const onednn_unary_key & other) const {
        return src_blob == other.src_blob &&
               op == other.op &&
               kernel == other.kernel &&
               stride == other.stride &&
               padding == other.padding;
    }
};

struct onednn_unary_key_hash {
    size_t operator()(const onednn_unary_key & key) const {
        size_t h = std::hash<std::string>{}(key.src_blob);
        h ^= std::hash<int>{}(key.op) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.kernel) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.stride) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.padding) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct onednn_conv_key {
    const float * weight_ptr = nullptr;
    const float * bias_ptr = nullptr;
    std::string src_blob;
    int input_w = 0;
    int input_h = 0;
    int out_c = 0;
    int in_c = 0;
    int kh = 0;
    int kw = 0;
    int stride = 0;
    int padding = 0;

    bool operator==(const onednn_conv_key & other) const {
        return weight_ptr == other.weight_ptr &&
               bias_ptr == other.bias_ptr &&
               src_blob == other.src_blob &&
               input_w == other.input_w &&
               input_h == other.input_h &&
               out_c == other.out_c &&
               in_c == other.in_c &&
               kh == other.kh &&
               kw == other.kw &&
               stride == other.stride &&
               padding == other.padding;
    }
};

struct onednn_conv_key_hash {
    size_t operator()(const onednn_conv_key & key) const {
        size_t h = std::hash<const void *>{}(key.weight_ptr);
        h ^= std::hash<const void *>{}(key.bias_ptr) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(key.src_blob) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.input_w) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.input_h) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.out_c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.in_c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.kh) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.kw) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.stride) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.padding) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct onednn_conv_plan {
    int out_w = 0;
    int out_h = 0;
    int out_c = 0;
    dnnl::convolution_forward::primitive_desc pd;
    dnnl::convolution_forward conv;
    dnnl::memory weights_memory;
    dnnl::memory bias_memory;

    onednn_conv_plan(
            int out_w,
            int out_h,
            int out_c,
            const dnnl::convolution_forward::primitive_desc & pd,
            const dnnl::convolution_forward & conv,
            const dnnl::memory & weights_memory,
            const dnnl::memory & bias_memory)
        : out_w(out_w),
          out_h(out_h),
          out_c(out_c),
          pd(pd),
          conv(conv),
          weights_memory(weights_memory),
          bias_memory(bias_memory) {}
};

struct onednn_eltwise_plan {
    dnnl::eltwise_forward::primitive_desc pd;
    dnnl::eltwise_forward prim;

    onednn_eltwise_plan(const dnnl::eltwise_forward::primitive_desc & pd, const dnnl::eltwise_forward & prim)
        : pd(pd), prim(prim) {}
};

struct onednn_pool_plan {
    int out_w = 0;
    int out_h = 0;
    dnnl::pooling_forward::primitive_desc pd;
    dnnl::pooling_forward prim;

    onednn_pool_plan(int out_w, int out_h, const dnnl::pooling_forward::primitive_desc & pd, const dnnl::pooling_forward & prim)
        : out_w(out_w), out_h(out_h), pd(pd), prim(prim) {}
};

struct onednn_sum_plan {
    dnnl::sum::primitive_desc pd;
    dnnl::sum prim;

    onednn_sum_plan(const dnnl::sum::primitive_desc & pd, const dnnl::sum & prim)
        : pd(pd), prim(prim) {}
};
#endif

svd_factors load_conv_svd_factors(
        ggml_context * weights_ctx,
        const std::string & weight_name,
        const ggml_tensor * weight);

int64_t require_key(const gguf_context * meta, const char * key) {
    const int64_t idx = gguf_find_key(meta, key);
    if (idx < 0) {
        throw std::runtime_error(std::string("missing GGUF key: ") + key);
    }
    return idx;
}

std::string get_string_key(const gguf_context * meta, const char * key) {
    return gguf_get_val_str(meta, require_key(meta, key));
}

uint32_t get_u32_key(const gguf_context * meta, const char * key, uint32_t fallback) {
    const int64_t idx = gguf_find_key(meta, key);
    if (idx < 0) {
        return fallback;
    }
    const gguf_type type = gguf_get_kv_type(meta, idx);
    if (type == GGUF_TYPE_UINT32) {
        return gguf_get_val_u32(meta, idx);
    }
    if (type == GGUF_TYPE_INT32) {
        return static_cast<uint32_t>(gguf_get_val_i32(meta, idx));
    }
    if (type == GGUF_TYPE_UINT64) {
        return static_cast<uint32_t>(gguf_get_val_u64(meta, idx));
    }
    if (type == GGUF_TYPE_INT64) {
        return static_cast<uint32_t>(gguf_get_val_i64(meta, idx));
    }
    throw std::runtime_error(std::string("unexpected scalar type for key: ") + key);
}

std::vector<float> get_float_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("key is not an array: ") + key);
    }
    const gguf_type arr_type = gguf_get_arr_type(meta, idx);
    const size_t n = gguf_get_arr_n(meta, idx);
    const void * raw = gguf_get_arr_data(meta, idx);
    std::vector<float> out(n);

    switch (arr_type) {
        case GGUF_TYPE_FLOAT32: {
            const auto * p = static_cast<const float *>(raw);
            std::copy(p, p + n, out.begin());
        } break;
        case GGUF_TYPE_FLOAT64: {
            const auto * p = static_cast<const double *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(p[i]);
            }
        } break;
        default:
            throw std::runtime_error(std::string("unexpected float array type for key: ") + key);
    }

    return out;
}

std::vector<uint32_t> get_u32_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("key is not an array: ") + key);
    }
    const gguf_type arr_type = gguf_get_arr_type(meta, idx);
    const size_t n = gguf_get_arr_n(meta, idx);
    const void * raw = gguf_get_arr_data(meta, idx);
    std::vector<uint32_t> out(n);

    switch (arr_type) {
        case GGUF_TYPE_UINT32: {
            const auto * p = static_cast<const uint32_t *>(raw);
            std::copy(p, p + n, out.begin());
        } break;
        case GGUF_TYPE_INT32: {
            const auto * p = static_cast<const int32_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        case GGUF_TYPE_UINT64: {
            const auto * p = static_cast<const uint64_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        case GGUF_TYPE_INT64: {
            const auto * p = static_cast<const int64_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        default:
            throw std::runtime_error(std::string("unexpected integer array type for key: ") + key);
    }

    return out;
}

std::vector<std::string> get_string_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("key is not a string array: ") + key);
    }
    const size_t n = gguf_get_arr_n(meta, idx);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(meta, idx, i));
    }
    return out;
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

ggml_tensor * optional_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

resnet50_model load_model(const std::string & model_path) {
    ggml_context * weights_ctx_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &weights_ctx_raw,
    };

    resnet50_model model;
    model.meta.reset(gguf_init_from_file(model_path.c_str(), params));
    if (!model.meta) {
        throw std::runtime_error("failed to load GGUF file: " + model_path);
    }
    model.weights.reset(weights_ctx_raw);
    if (!model.weights) {
        throw std::runtime_error("failed to materialize GGUF tensor context");
    }

    const std::string arch = get_string_key(model.meta.get(), "general.architecture");
    if (arch != "resnet50") {
        throw std::runtime_error("unsupported architecture: " + arch);
    }

    model.preproc.image_size = get_u32_key(model.meta.get(), "resnet50.image_size", model.preproc.image_size);
    model.preproc.resize_shorter_edge = get_u32_key(model.meta.get(), "resnet50.resize_shorter_edge", model.preproc.resize_shorter_edge);

    const auto mean = get_float_array(model.meta.get(), "resnet50.input_mean");
    const auto stdv = get_float_array(model.meta.get(), "resnet50.input_std");
    if (mean.size() != 3 || stdv.size() != 3) {
        throw std::runtime_error("expected 3-channel normalization metadata");
    }
    for (int i = 0; i < 3; ++i) {
        model.preproc.mean[i] = mean[i];
        model.preproc.std[i] = stdv[i];
    }

    const auto stages = get_u32_array(model.meta.get(), "resnet50.stage_block_count");
    if (stages.size() != 4) {
        throw std::runtime_error("expected 4 stage block counts");
    }
    for (int i = 0; i < 4; ++i) {
        model.stage_block_count[i] = stages[i];
    }

    model.labels = get_string_array(model.meta.get(), "resnet50.labels");

    return model;
}

image_u8 load_image_rgb(const std::string & image_path) {
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char * raw = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (raw == nullptr) {
        throw std::runtime_error("failed to load image: " + image_path);
    }

    image_u8 image;
    image.width = w;
    image.height = h;
    image.channels = 3;
    image.data.assign(raw, raw + (w * h * 3));
    stbi_image_free(raw);
    return image;
}

std::vector<uint8_t> resize_bilinear_rgb(const image_u8 & image, int dst_w, int dst_h) {
    std::vector<uint8_t> out(dst_w * dst_h * 3);
    const float scale_x = static_cast<float>(image.width) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(image.height) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float ly = src_y - static_cast<float>(y0);

        for (int x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float lx = src_x - static_cast<float>(x0);

            for (int ch = 0; ch < 3; ++ch) {
                const float p00 = image.data[(y0 * image.width + x0) * 3 + ch];
                const float p01 = image.data[(y0 * image.width + x1) * 3 + ch];
                const float p10 = image.data[(y1 * image.width + x0) * 3 + ch];
                const float p11 = image.data[(y1 * image.width + x1) * 3 + ch];

                const float top = p00 + (p01 - p00) * lx;
                const float bottom = p10 + (p11 - p10) * lx;
                const float value = top + (bottom - top) * ly;

                out[(y * dst_w + x) * 3 + ch] = static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l));
            }
        }
    }

    return out;
}

std::vector<float> preprocess_image(const image_u8 & image, const preproc_config & cfg) {
    const int resize_short = static_cast<int>(cfg.resize_shorter_edge);
    const int crop = static_cast<int>(cfg.image_size);

    int resized_w = 0;
    int resized_h = 0;
    if (image.width < image.height) {
        resized_w = resize_short;
        resized_h = static_cast<int>(std::round(static_cast<float>(image.height) * resize_short / image.width));
    } else {
        resized_h = resize_short;
        resized_w = static_cast<int>(std::round(static_cast<float>(image.width) * resize_short / image.height));
    }

    const std::vector<uint8_t> resized = resize_bilinear_rgb(image, resized_w, resized_h);
    const int x0 = std::max(0, (resized_w - crop) / 2);
    const int y0 = std::max(0, (resized_h - crop) / 2);

    std::vector<float> out(crop * crop * 3);
    for (int y = 0; y < crop; ++y) {
        for (int x = 0; x < crop; ++x) {
            const int src_x = x0 + x;
            const int src_y = y0 + y;
            for (int ch = 0; ch < 3; ++ch) {
                const float pixel = resized[(src_y * resized_w + src_x) * 3 + ch] / 255.0f;
                const float normalized = (pixel - cfg.mean[ch]) / cfg.std[ch];
                out[x + crop * (y + crop * ch)] = normalized;
            }
        }
    }

    return out;
}

template <typename Fn>
void parallel_for_chunks(int total, int n_threads, Fn fn) {
    const int nth = std::max(1, std::min(total, n_threads));
    if (nth == 1) {
        fn(0, total);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nth);
    for (int tid = 0; tid < nth; ++tid) {
        const int begin = total * tid / nth;
        const int end = total * (tid + 1) / nth;
        workers.emplace_back([=, &fn]() {
            fn(begin, end);
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
}

float tensor_scalar(const ggml_tensor * tensor, int64_t idx) {
    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float *>(tensor->data)[idx];
    }
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(tensor->data)[idx]);
    }
    throw std::runtime_error("unsupported tensor type");
}

float bias_value(const ggml_tensor * bias, int channel) {
    return tensor_scalar(bias, channel);
}

#ifdef RESNET50_USE_ONEDNN
dnnl::engine & onednn_engine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

dnnl::stream & onednn_stream() {
    static dnnl::stream s(onednn_engine());
    return s;
}

dnnl::memory::desc nchw_desc(const feature_map & fm) {
    return dnnl::memory::desc({1, fm.channels, fm.height, fm.width}, dt::f32, tag::nchw);
}

void bind_onednn_memory(feature_map & fm, const dnnl::memory::desc & md) {
    fm.onednn_buffer.resize(md.get_size());
    fm.onednn_memory = dnnl::memory(md, onednn_engine(), fm.onednn_buffer.data());
    fm.onednn_valid = true;
}

void ensure_plain(feature_map & fm) {
    if (fm.data_valid) {
        return;
    }
    const dnnl::memory::desc user_md = nchw_desc(fm);
    fm.data.resize(static_cast<size_t>(fm.width * fm.height * fm.channels));
    dnnl::memory user_mem(user_md, onednn_engine(), fm.data.data());
    if (fm.onednn_memory.get_desc() == user_md) {
        std::memcpy(fm.data.data(), fm.onednn_buffer.data(), sizeof(float) * fm.data.size());
    } else {
        dnnl::reorder(fm.onednn_memory, user_mem).execute(onednn_stream(), fm.onednn_memory, user_mem);
        onednn_stream().wait();
    }
    fm.data_valid = true;
}

void ensure_onednn(feature_map & fm, const dnnl::memory::desc * target_md = nullptr) {
    if (!fm.onednn_valid) {
        const dnnl::memory::desc user_md = nchw_desc(fm);
        bind_onednn_memory(fm, user_md);
        std::memcpy(fm.onednn_buffer.data(), fm.data.data(), sizeof(float) * fm.data.size());
        fm.onednn_valid = true;
    }

    if (target_md != nullptr && fm.onednn_memory.get_desc() != *target_md) {
        std::vector<uint8_t> new_buffer(target_md->get_size());
        dnnl::memory new_mem(*target_md, onednn_engine(), new_buffer.data());
        dnnl::reorder(fm.onednn_memory, new_mem).execute(onednn_stream(), fm.onednn_memory, new_mem);
        onednn_stream().wait();
        fm.onednn_buffer = std::move(new_buffer);
        fm.onednn_memory = new_mem;
    }

    fm.onednn_valid = true;
    fm.data_valid = false;
}

const onednn_eltwise_plan & get_onednn_relu_plan(const dnnl::memory::desc & src_md) {
    static std::unordered_map<onednn_unary_key, onednn_eltwise_plan, onednn_unary_key_hash> cache;
    onednn_unary_key key{blob_key(src_md), 1, 0, 0, 0};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    const auto pd = dnnl::eltwise_forward::primitive_desc(
            onednn_engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu, src_md, src_md, 0.0f);
    auto inserted = cache.emplace(key, onednn_eltwise_plan(pd, dnnl::eltwise_forward(pd)));
    return inserted.first->second;
}

void relu_onednn_inplace(feature_map & fm) {
    ensure_onednn(fm);
    const onednn_eltwise_plan & plan = get_onednn_relu_plan(fm.onednn_memory.get_desc());
    feature_map out;
    out.width = fm.width;
    out.height = fm.height;
    out.channels = fm.channels;
    bind_onednn_memory(out, fm.onednn_memory.get_desc());
    plan.prim.execute(onednn_stream(), {{DNNL_ARG_SRC, fm.onednn_memory}, {DNNL_ARG_DST, out.onednn_memory}});
    onednn_stream().wait();
    out.data_valid = false;
    fm = std::move(out);
}

const onednn_pool_plan & get_onednn_max_pool_plan(
        const dnnl::memory::desc & src_md,
        int width,
        int height,
        int channels,
        int kernel,
        int stride,
        int padding) {
    static std::unordered_map<onednn_unary_key, onednn_pool_plan, onednn_unary_key_hash> cache;
    onednn_unary_key key{blob_key(src_md), 2, kernel, stride, padding};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    const int out_w = (width + 2 * padding - kernel) / stride + 1;
    const int out_h = (height + 2 * padding - kernel) / stride + 1;
    const auto dst_md = dnnl::memory::desc({1, channels, out_h, out_w}, dt::f32, tag::any);
    const auto pd = dnnl::pooling_forward::primitive_desc(
            onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_max,
            src_md,
            dst_md,
            {stride, stride},
            {kernel, kernel},
            {0, 0},
            {padding, padding},
            {padding, padding});
    auto inserted = cache.emplace(key, onednn_pool_plan(out_w, out_h, pd, dnnl::pooling_forward(pd)));
    return inserted.first->second;
}

feature_map max_pool_2d_onednn(feature_map & input, int kernel, int stride, int padding) {
    ensure_onednn(input);
    const onednn_pool_plan & plan = get_onednn_max_pool_plan(
            input.onednn_memory.get_desc(), input.width, input.height, input.channels, kernel, stride, padding);
    feature_map out;
    out.width = plan.out_w;
    out.height = plan.out_h;
    out.channels = input.channels;
    bind_onednn_memory(out, plan.pd.dst_desc());
    const dnnl::memory::desc src_desc = plan.pd.src_desc();
    ensure_onednn(input, &src_desc);
    plan.prim.execute(onednn_stream(), {{DNNL_ARG_SRC, input.onednn_memory}, {DNNL_ARG_DST, out.onednn_memory}});
    onednn_stream().wait();
    out.data_valid = false;
    return out;
}

const onednn_sum_plan & get_onednn_sum_plan(const dnnl::memory::desc & desc) {
    static std::unordered_map<std::string, onednn_sum_plan> cache;
    const std::string key = blob_key(desc);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<float> scales = {1.0f, 1.0f};
    std::vector<dnnl::memory::desc> srcs = {desc, desc};
    const auto pd = dnnl::sum::primitive_desc(onednn_engine(), scales, srcs);
    auto inserted = cache.emplace(key, onednn_sum_plan(pd, dnnl::sum(pd)));
    return inserted.first->second;
}

void add_onednn_inplace(feature_map & dst, feature_map & src) {
    ensure_onednn(dst);
    const dnnl::memory::desc dst_desc = dst.onednn_memory.get_desc();
    ensure_onednn(src, &dst_desc);
    const onednn_sum_plan & plan = get_onednn_sum_plan(dst_desc);
    feature_map out;
    out.width = dst.width;
    out.height = dst.height;
    out.channels = dst.channels;
    bind_onednn_memory(out, dst_desc);
    plan.prim.execute(
            onednn_stream(),
            {{DNNL_ARG_MULTIPLE_SRC + 0, dst.onednn_memory},
             {DNNL_ARG_MULTIPLE_SRC + 1, src.onednn_memory},
             {DNNL_ARG_DST, out.onednn_memory}});
    onednn_stream().wait();
    out.data_valid = false;
    dst = std::move(out);
}

struct onednn_weight_bundle {
    std::vector<float> weight_data;
    std::vector<float> bias_data;
};

onednn_weight_bundle make_onednn_conv_bundle_from_svd(
        const svd_factors & factors,
        int kh,
        int kw,
        int in_channels,
        int out_channels,
        bool use_u,
        const ggml_tensor * bias) {
    onednn_weight_bundle out;
    if (use_u) {
        out.weight_data.resize(static_cast<size_t>(out_channels * factors.rank));
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < factors.rank; ++ic) {
                out.weight_data[static_cast<size_t>(ic + factors.rank * oc)] = factors.u[static_cast<size_t>(ic + factors.rank * oc)];
            }
        }
        out.bias_data.resize(static_cast<size_t>(out_channels), 0.0f);
        for (int oc = 0; oc < out_channels; ++oc) {
            out.bias_data[static_cast<size_t>(oc)] = bias ? bias_value(bias, oc) : 0.0f;
        }
        return out;
    }

    out.weight_data.resize(static_cast<size_t>(out_channels * in_channels * kh * kw));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    const int64_t row = kx + static_cast<int64_t>(kw) * (ky + kh * ic);
                    out.weight_data[static_cast<size_t>(kx + kw * (ky + kh * (ic + in_channels * oc)))] =
                            factors.v[static_cast<size_t>(row + factors.input_len * oc)];
                }
            }
        }
    }
    out.bias_data.assign(static_cast<size_t>(out_channels), 0.0f);
    return out;
}

const onednn_conv_plan & get_onednn_conv_plan_raw(
        const float * weight_ptr,
        const float * bias_ptr,
        const dnnl::memory::desc & src_md,
        int input_w,
        int input_h,
        int out_c,
        int in_c,
        int kh,
        int kw,
        int stride,
        int padding) {
    static std::unordered_map<onednn_conv_key, onednn_conv_plan, onednn_conv_key_hash> cache;

    onednn_conv_key key{weight_ptr, bias_ptr, blob_key(src_md), input_w, input_h, out_c, in_c, kh, kw, stride, padding};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int out_w = (input_w + 2 * padding - kw) / stride + 1;
    const int out_h = (input_h + 2 * padding - kh) / stride + 1;
    const dnnl::memory::dims weight_dims = {out_c, in_c, kh, kw};
    const dnnl::memory::dims bias_dims = {out_c};
    const dnnl::memory::dims strides = {stride, stride};
    const dnnl::memory::dims pads = {padding, padding};

    const auto user_weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::oihw);
    const auto bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::x);
    const auto dst_md = dnnl::memory::desc({1, out_c, out_h, out_w}, dt::f32, tag::any);
    const auto conv_pd = dnnl::convolution_forward::primitive_desc(
            onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            src_md,
            dnnl::memory::desc(weight_dims, dt::f32, tag::any),
            bias_md,
            dst_md,
            strides,
            pads,
            pads);

    dnnl::memory user_weight_memory(user_weight_md, onednn_engine(), const_cast<float *>(weight_ptr));
    dnnl::memory weight_memory(conv_pd.weights_desc(), onednn_engine());
    if (conv_pd.weights_desc() != user_weight_md) {
        dnnl::reorder(user_weight_memory, weight_memory).execute(onednn_stream(), user_weight_memory, weight_memory);
        onednn_stream().wait();
    } else {
        weight_memory = user_weight_memory;
    }

    dnnl::memory bias_memory(bias_md, onednn_engine(), const_cast<float *>(bias_ptr));
    dnnl::convolution_forward conv(conv_pd);
    auto inserted = cache.emplace(key, onednn_conv_plan(out_w, out_h, out_c, conv_pd, conv, weight_memory, bias_memory));
    return inserted.first->second;
}

feature_map conv2d_svd_fold_onednn(
        ggml_context * weights_ctx,
        const std::string & base,
        feature_map & input,
        int stride,
        int padding) {
    const std::string weight_name = base + ".weight";
    const std::string bias_name = base + ".bias";
    const ggml_tensor * weight = require_tensor(weights_ctx, weight_name);
    const ggml_tensor * bias = require_tensor(weights_ctx, bias_name);
    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);

    static std::unordered_map<std::string, svd_factors> factor_cache;
    static std::unordered_map<std::string, onednn_weight_bundle> weight_cache;
    if (factor_cache.find(weight_name) == factor_cache.end()) {
        factor_cache.emplace(weight_name, load_conv_svd_factors(weights_ctx, weight_name, weight));
    }
    const svd_factors & factors = factor_cache.at(weight_name);

    const std::string key_v = weight_name + "#v";
    const std::string key_u = weight_name + "#u";
    if (weight_cache.find(key_v) == weight_cache.end()) {
        weight_cache.emplace(key_v, make_onednn_conv_bundle_from_svd(factors, kh, kw, in_channels, static_cast<int>(factors.rank), false, nullptr));
    }
    if (weight_cache.find(key_u) == weight_cache.end()) {
        weight_cache.emplace(key_u, make_onednn_conv_bundle_from_svd(factors, 1, 1, static_cast<int>(factors.rank), out_channels, true, bias));
    }
    const onednn_weight_bundle & v_bundle = weight_cache.at(key_v);
    const onednn_weight_bundle & u_bundle = weight_cache.at(key_u);

    ensure_onednn(input);
    const dnnl::memory::desc input_desc = input.onednn_memory.get_desc();
    const onednn_conv_plan & v_plan = get_onednn_conv_plan_raw(
            v_bundle.weight_data.data(),
            v_bundle.bias_data.data(),
            input_desc,
            input.width,
            input.height,
            static_cast<int>(factors.rank),
            in_channels,
            kh,
            kw,
            stride,
            padding);
    const dnnl::memory::desc v_src_desc = v_plan.pd.src_desc();
    ensure_onednn(input, &v_src_desc);

    feature_map rank_map;
    rank_map.width = v_plan.out_w;
    rank_map.height = v_plan.out_h;
    rank_map.channels = static_cast<int>(factors.rank);
    bind_onednn_memory(rank_map, v_plan.pd.dst_desc());
    v_plan.conv.execute(
            onednn_stream(),
            {{DNNL_ARG_SRC, input.onednn_memory},
             {DNNL_ARG_WEIGHTS, v_plan.weights_memory},
             {DNNL_ARG_BIAS, v_plan.bias_memory},
             {DNNL_ARG_DST, rank_map.onednn_memory}});
    onednn_stream().wait();
    rank_map.data_valid = false;

    const onednn_conv_plan & u_plan = get_onednn_conv_plan_raw(
            u_bundle.weight_data.data(),
            u_bundle.bias_data.data(),
            rank_map.onednn_memory.get_desc(),
            rank_map.width,
            rank_map.height,
            out_channels,
            rank_map.channels,
            1,
            1,
            1,
            0);
    const dnnl::memory::desc u_src_desc = u_plan.pd.src_desc();
    ensure_onednn(rank_map, &u_src_desc);

    feature_map out;
    out.width = u_plan.out_w;
    out.height = u_plan.out_h;
    out.channels = out_channels;
    bind_onednn_memory(out, u_plan.pd.dst_desc());
    u_plan.conv.execute(
            onednn_stream(),
            {{DNNL_ARG_SRC, rank_map.onednn_memory},
             {DNNL_ARG_WEIGHTS, u_plan.weights_memory},
             {DNNL_ARG_BIAS, u_plan.bias_memory},
             {DNNL_ARG_DST, out.onednn_memory}});
    onednn_stream().wait();
    out.data_valid = false;
    return out;
}
#endif

std::string make_svd_tensor_name(const std::string & weight_name, const char * suffix) {
    return weight_name + suffix;
}

std::vector<float> make_identity_matrix(int64_t n) {
    std::vector<float> out(static_cast<size_t>(n * n), 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i + n * i)] = 1.0f;
    }
    return out;
}

std::vector<float> extract_tensor_data_f32(const ggml_tensor * tensor) {
    const int64_t count = ggml_nelements(tensor);
    std::vector<float> out(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] = tensor_scalar(tensor, i);
    }
    return out;
}

std::vector<float> unfold_conv_weight_to_matrix(const ggml_tensor * weight) {
    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    const int64_t input_len = static_cast<int64_t>(kw) * kh * in_channels;
    std::vector<float> out(static_cast<size_t>(input_len * out_channels));

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            const int64_t w_base = static_cast<int64_t>(kw) * kh * (ic + static_cast<int64_t>(in_channels) * oc);
            const int64_t in_base = static_cast<int64_t>(kw) * kh * ic;
            for (int ky = 0; ky < kh; ++ky) {
                const int64_t w_row = w_base + static_cast<int64_t>(kw) * ky;
                const int64_t in_row = in_base + static_cast<int64_t>(kw) * ky;
                for (int kx = 0; kx < kw; ++kx) {
                    const int64_t idx = in_row + kx + input_len * oc;
                    out[static_cast<size_t>(idx)] = tensor_scalar(weight, w_row + kx);
                }
            }
        }
    }

    return out;
}

svd_factors load_conv_svd_factors(
        ggml_context * weights_ctx,
        const std::string & weight_name,
        const ggml_tensor * weight) {
    const int64_t input_len = weight->ne[0] * weight->ne[1] * weight->ne[2];
    const int64_t output_len = weight->ne[3];

    const std::string v_name = make_svd_tensor_name(weight_name, "_svd_v");
    const std::string u_name = make_svd_tensor_name(weight_name, "_svd_u");

    if (ggml_tensor * tensor_v = optional_tensor(weights_ctx, v_name)) {
        ggml_tensor * tensor_u = require_tensor(weights_ctx, u_name);
        if (tensor_v->ne[0] != input_len) {
            throw std::runtime_error("unexpected conv SVD V input size for tensor: " + v_name);
        }
        if (tensor_u->ne[1] != output_len) {
            throw std::runtime_error("unexpected conv SVD U output size for tensor: " + u_name);
        }
        if (tensor_v->ne[1] != tensor_u->ne[0]) {
            throw std::runtime_error("conv SVD rank mismatch between tensors: " + v_name + " and " + u_name);
        }

        svd_factors out;
        out.input_len = input_len;
        out.rank = tensor_v->ne[1];
        out.output_len = output_len;
        out.v = extract_tensor_data_f32(tensor_v);
        out.u = extract_tensor_data_f32(tensor_u);
        out.from_precomputed_svd = true;
        return out;
    }

    svd_factors out;
    out.input_len = input_len;
    out.rank = output_len;
    out.output_len = output_len;
    out.v = unfold_conv_weight_to_matrix(weight);
    out.u = make_identity_matrix(output_len);
    out.from_precomputed_svd = false;
    return out;
}

std::vector<float> unfold_input_im2col(
        const feature_map & input,
        int kw,
        int kh,
        int stride,
        int padding,
        int out_w,
        int out_h) {
    const int64_t input_len = static_cast<int64_t>(kw) * kh * input.channels;
    const int64_t n_cols = static_cast<int64_t>(out_w) * out_h;
    std::vector<float> cols(static_cast<size_t>(input_len * n_cols), 0.0f);

    for (int oy = 0; oy < out_h; ++oy) {
        const int iy_base = oy * stride - padding;
        for (int ox = 0; ox < out_w; ++ox) {
            const int ix_base = ox * stride - padding;
            const int64_t col = ox + static_cast<int64_t>(out_w) * oy;
            for (int ic = 0; ic < input.channels; ++ic) {
                const int64_t channel_base = static_cast<int64_t>(kw) * kh * ic;
                for (int ky = 0; ky < kh; ++ky) {
                    const int iy = iy_base + ky;
                    if (iy < 0 || iy >= input.height) {
                        continue;
                    }
                    const int64_t row_base = channel_base + static_cast<int64_t>(kw) * ky;
                    for (int kx = 0; kx < kw; ++kx) {
                        const int ix = ix_base + kx;
                        if (ix < 0 || ix >= input.width) {
                            continue;
                        }
                        const int64_t row = row_base + kx;
                        cols[static_cast<size_t>(row + input_len * col)] = input.at(ic, iy, ix);
                    }
                }
            }
        }
    }

    return cols;
}

std::vector<float> eval_mul_mat_svd(
        const svd_factors & factors,
        const std::vector<float> & input_cols,
        int64_t n_cols,
        int n_threads) {
    if (static_cast<int64_t>(input_cols.size()) != factors.input_len * n_cols) {
        throw std::runtime_error("mul_mat_svd input size mismatch");
    }

    const size_t bytes =
            8 * 1024 * 1024 +
            sizeof(float) * (
                    static_cast<size_t>(factors.input_len * n_cols) +
                    factors.v.size() +
                    factors.u.size() +
                    static_cast<size_t>(factors.input_len * factors.output_len) +
                    static_cast<size_t>(factors.output_len * n_cols));

    ggml_init_params params = {
        /*.mem_size   =*/ bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_ctx_ptr ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for conv mul_mat_svd");
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, factors.input_len, n_cols);
    ggml_tensor * w_shape = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, factors.input_len, factors.output_len);
    ggml_tensor * v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, factors.input_len, factors.rank);
    ggml_tensor * u = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, factors.rank, factors.output_len);
    if (input == nullptr || w_shape == nullptr || v == nullptr || u == nullptr) {
        throw std::runtime_error("failed to allocate ggml tensors for conv mul_mat_svd");
    }

    std::memcpy(input->data, input_cols.data(), sizeof(float) * input_cols.size());
    std::memcpy(v->data, factors.v.data(), sizeof(float) * factors.v.size());
    std::memcpy(u->data, factors.u.data(), sizeof(float) * factors.u.size());

    ggml_tensor * output = ggml_mul_mat_svd(ctx.get(), w_shape, v, u, input, 0);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    if (graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed for conv mul_mat_svd");
    }

    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute_with_ctx failed for conv mul_mat_svd");
    }

    const size_t output_count = static_cast<size_t>(factors.output_len * n_cols);
    const float * output_ptr = static_cast<const float *>(output->data);
    return std::vector<float>(output_ptr, output_ptr + output_count);
}

std::vector<float> eval_mul_mat_f32(
        const std::vector<float> & weight,
        int64_t input_len,
        int64_t output_len,
        const std::vector<float> & input_cols,
        int64_t n_cols,
        int n_threads) {
    if (static_cast<int64_t>(weight.size()) != input_len * output_len) {
        throw std::runtime_error("mul_mat weight size mismatch");
    }
    if (static_cast<int64_t>(input_cols.size()) != input_len * n_cols) {
        throw std::runtime_error("mul_mat input size mismatch");
    }

    const size_t bytes =
            8 * 1024 * 1024 +
            sizeof(float) * (
                    weight.size() +
                    input_cols.size() +
                    static_cast<size_t>(output_len * n_cols));

    ggml_init_params params = {
        /*.mem_size   =*/ bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_ctx_ptr ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for conv mul_mat");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, input_len, output_len);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, input_len, n_cols);
    if (w == nullptr || input == nullptr) {
        throw std::runtime_error("failed to allocate ggml tensors for conv mul_mat");
    }

    std::memcpy(w->data, weight.data(), sizeof(float) * weight.size());
    std::memcpy(input->data, input_cols.data(), sizeof(float) * input_cols.size());

    ggml_tensor * output = ggml_mul_mat(ctx.get(), w, input);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    if (graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed for conv mul_mat");
    }

    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute_with_ctx failed for conv mul_mat");
    }

    const size_t output_count = static_cast<size_t>(output_len * n_cols);
    const float * output_ptr = static_cast<const float *>(output->data);
    return std::vector<float>(output_ptr, output_ptr + output_count);
}

feature_map output_cols_to_feature_map(
        const std::vector<float> & output_cols,
        int width,
        int height,
        int channels,
        const ggml_tensor * bias,
        int n_threads) {
    const int64_t plane = static_cast<int64_t>(width) * height;
    if (static_cast<int64_t>(output_cols.size()) != static_cast<int64_t>(channels) * plane) {
        throw std::runtime_error("output column size mismatch");
    }

    feature_map out;
    out.width = width;
    out.height = height;
    out.channels = channels;
    out.data.assign(static_cast<size_t>(plane * channels), 0.0f);

    parallel_for_chunks(channels, n_threads, [&](int c_begin, int c_end) {
        for (int c = c_begin; c < c_end; ++c) {
            const float b = bias ? bias_value(bias, c) : 0.0f;
            const size_t dst_base = static_cast<size_t>(plane * c);
            for (int64_t pos = 0; pos < plane; ++pos) {
                out.data[dst_base + static_cast<size_t>(pos)] =
                        output_cols[static_cast<size_t>(c + static_cast<int64_t>(channels) * pos)] + b;
            }
        }
    });

    return out;
}

feature_map conv2d_matrixized_svd(
        ggml_context * weights_ctx,
        const std::string & base,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads) {
    const std::string weight_name = base + ".weight";
    const std::string bias_name = base + ".bias";
    const ggml_tensor * weight = require_tensor(weights_ctx, weight_name);
    const ggml_tensor * bias = require_tensor(weights_ctx, bias_name);

    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv input channel mismatch");
    }

    feature_map out;
    out.width = (input.width + 2 * padding - kw) / stride + 1;
    out.height = (input.height + 2 * padding - kh) / stride + 1;
    out.channels = out_channels;
    out.data.assign(static_cast<size_t>(out.width * out.height * out.channels), 0.0f);

    const int64_t out_plane = static_cast<int64_t>(out.width) * out.height;
    const std::vector<float> input_cols = unfold_input_im2col(input, kw, kh, stride, padding, out.width, out.height);
    const svd_factors factors = load_conv_svd_factors(weights_ctx, weight_name, weight);
    if (factors.input_len != static_cast<int64_t>(kw) * kh * in_channels || factors.output_len != out_channels) {
        throw std::runtime_error("conv SVD factor shape mismatch");
    }

    const std::vector<float> output_cols = eval_mul_mat_svd(factors, input_cols, out_plane, n_threads);

    parallel_for_chunks(out_channels, n_threads, [&](int oc_begin, int oc_end) {
        for (int oc = oc_begin; oc < oc_end; ++oc) {
            const float b = bias_value(bias, oc);
            for (int oy = 0; oy < out.height; ++oy) {
                for (int ox = 0; ox < out.width; ++ox) {
                    const int64_t col = ox + static_cast<int64_t>(out.width) * oy;
                    const float value = output_cols[static_cast<size_t>(oc + factors.output_len * col)] + b;
                    out.data[static_cast<size_t>(ox + out.width * (oy + out.height * oc))] = value;
                }
            }
        }
    });

    return out;
}

rank_cols_result folded_v_conv_im2col_ggml_cols(
        const feature_map & input,
        const svd_factors & factors,
        int kw,
        int kh,
        int stride,
        int padding,
        int n_threads) {
    rank_cols_result out;
    out.width = (input.width + 2 * padding - kw) / stride + 1;
    out.height = (input.height + 2 * padding - kh) / stride + 1;
    out.channels = static_cast<int>(factors.rank);

    const int64_t n_cols = static_cast<int64_t>(out.width) * out.height;
    const std::vector<float> input_cols = unfold_input_im2col(input, kw, kh, stride, padding, out.width, out.height);
    out.cols = eval_mul_mat_f32(
            factors.v,
            factors.input_len,
            factors.rank,
            input_cols,
            n_cols,
            n_threads);

    return out;
}

feature_map conv2d_fold_svd(
        ggml_context * weights_ctx,
        const std::string & base,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads) {
#ifdef RESNET50_USE_ONEDNN
    (void)n_threads;
    return conv2d_svd_fold_onednn(weights_ctx, base, const_cast<feature_map &>(input), stride, padding);
#endif
    const std::string weight_name = base + ".weight";
    const std::string bias_name = base + ".bias";
    const ggml_tensor * weight = require_tensor(weights_ctx, weight_name);
    const ggml_tensor * bias = require_tensor(weights_ctx, bias_name);

    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv input channel mismatch");
    }

    const svd_factors factors = load_conv_svd_factors(weights_ctx, weight_name, weight);
    if (factors.input_len != static_cast<int64_t>(kw) * kh * in_channels || factors.output_len != out_channels) {
        throw std::runtime_error("conv SVD factor shape mismatch");
    }

    rank_cols_result rank_result = folded_v_conv_im2col_ggml_cols(input, factors, kw, kh, stride, padding, n_threads);

    const std::vector<float> output_cols = eval_mul_mat_f32(
            factors.u,
            factors.rank,
            factors.output_len,
            rank_result.cols,
            static_cast<int64_t>(rank_result.width) * rank_result.height,
            n_threads);

    return output_cols_to_feature_map(output_cols, rank_result.width, rank_result.height, out_channels, bias, n_threads);
}

feature_map conv2d_svd(
        ggml_context * weights_ctx,
        const std::string & base,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads,
        const std::string & mode) {
    if (mode == "im2col") {
        return conv2d_matrixized_svd(weights_ctx, base, input, stride, padding, n_threads);
    }
    if (mode == "fold") {
        return conv2d_fold_svd(weights_ctx, base, input, stride, padding, n_threads);
    }
    throw std::runtime_error("unknown conv SVD mode: " + mode);
}

void relu_inplace(feature_map & fm) {
#ifdef RESNET50_USE_ONEDNN
    relu_onednn_inplace(fm);
    return;
#endif
    for (float & v : fm.data) {
        v = std::max(0.0f, v);
    }
}

feature_map max_pool_2d(const feature_map & input, int kernel, int stride, int padding) {
#ifdef RESNET50_USE_ONEDNN
    return max_pool_2d_onednn(const_cast<feature_map &>(input), kernel, stride, padding);
#endif
    feature_map out;
    out.width = (input.width + 2 * padding - kernel) / stride + 1;
    out.height = (input.height + 2 * padding - kernel) / stride + 1;
    out.channels = input.channels;
    out.data.assign(static_cast<size_t>(out.width * out.height * out.channels), 0.0f);

    const int out_plane = out.width * out.height;
    for (int c = 0; c < out.channels; ++c) {
        for (int oy = 0; oy < out.height; ++oy) {
            const int iy_base = oy * stride - padding;
            for (int ox = 0; ox < out.width; ++ox) {
                const int ix_base = ox * stride - padding;
                float best = -std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < kernel; ++ky) {
                    const int iy = iy_base + ky;
                    if (iy < 0 || iy >= input.height) {
                        continue;
                    }
                    for (int kx = 0; kx < kernel; ++kx) {
                        const int ix = ix_base + kx;
                        if (ix < 0 || ix >= input.width) {
                            continue;
                        }
                        best = std::max(best, input.at(c, iy, ix));
                    }
                }
                out.data[static_cast<size_t>(ox + out.width * oy + out_plane * c)] = best;
            }
        }
    }
    return out;
}

void add_inplace(feature_map & dst, const feature_map & src) {
    if (dst.width != src.width || dst.height != src.height || dst.channels != src.channels) {
        throw std::runtime_error("residual shape mismatch");
    }
#ifdef RESNET50_USE_ONEDNN
    add_onednn_inplace(dst, const_cast<feature_map &>(src));
    return;
#endif
    for (size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] += src.data[i];
    }
}

feature_map bottleneck_block(
        const resnet50_model & model,
        const feature_map & input,
        int stage_idx,
        int block_idx,
        int n_threads,
        const std::string & mode) {
    const std::string base = "resnet.stage." + std::to_string(stage_idx) + ".block." + std::to_string(block_idx);

    feature_map identity = input;
    if (optional_tensor(model.weights.get(), base + ".downsample.weight") != nullptr) {
        identity = conv2d_svd(
                model.weights.get(),
                base + ".downsample",
                identity,
                block_idx == 0 && stage_idx > 0 ? 2 : 1,
                0,
                n_threads,
                mode);
    }

    feature_map out = conv2d_svd(
            model.weights.get(),
            base + ".conv1",
            input,
            1,
            0,
            n_threads,
            mode);
    relu_inplace(out);

    out = conv2d_svd(
            model.weights.get(),
            base + ".conv2",
            out,
            block_idx == 0 && stage_idx > 0 ? 2 : 1,
            1,
            n_threads,
            mode);
    relu_inplace(out);

    out = conv2d_svd(
            model.weights.get(),
            base + ".conv3",
            out,
            1,
            0,
            n_threads,
            mode);

    add_inplace(out, identity);
    relu_inplace(out);
    return out;
}

std::vector<float> run_inference(
        const resnet50_model & model,
        const std::vector<float> & input,
        int n_threads,
        const std::string & mode) {
    const int image_size = static_cast<int>(model.preproc.image_size);
    feature_map cur;
    cur.width = image_size;
    cur.height = image_size;
    cur.channels = 3;
    cur.data = input;

    cur = conv2d_svd(
            model.weights.get(),
            "resnet.stem.conv",
            cur,
            2,
            3,
            n_threads,
            mode);
    relu_inplace(cur);
    cur = max_pool_2d(cur, 3, 2, 1);

    for (int stage_idx = 0; stage_idx < 4; ++stage_idx) {
        for (uint32_t block_idx = 0; block_idx < model.stage_block_count[stage_idx]; ++block_idx) {
            cur = bottleneck_block(model, cur, stage_idx, static_cast<int>(block_idx), n_threads, mode);
        }
    }

#ifdef RESNET50_USE_ONEDNN
    ensure_plain(cur);
#endif
    std::vector<float> pooled(static_cast<size_t>(cur.channels), 0.0f);
    const int spatial = cur.width * cur.height;
    for (int c = 0; c < cur.channels; ++c) {
        double sum = 0.0;
        for (int y = 0; y < cur.height; ++y) {
            for (int x = 0; x < cur.width; ++x) {
                sum += cur.at(c, y, x);
            }
        }
        pooled[c] = static_cast<float>(sum / spatial);
    }

    const ggml_tensor * classifier_w = require_tensor(model.weights.get(), "resnet.classifier.weight");
    const ggml_tensor * classifier_b = require_tensor(model.weights.get(), "resnet.classifier.bias");
    const int in_features = static_cast<int>(classifier_w->ne[0]);
    const int out_features = static_cast<int>(classifier_w->ne[1]);
    if (in_features != cur.channels) {
        throw std::runtime_error("classifier input size mismatch");
    }

    std::vector<float> logits(static_cast<size_t>(out_features), 0.0f);
    parallel_for_chunks(out_features, n_threads, [&](int begin, int end) {
        for (int oc = begin; oc < end; ++oc) {
            float sum = tensor_scalar(classifier_b, oc);
            const int64_t w_base = static_cast<int64_t>(in_features) * oc;
            for (int ic = 0; ic < in_features; ++ic) {
                sum += pooled[ic] * tensor_scalar(classifier_w, w_base + ic);
            }
            logits[oc] = sum;
        }
    });

    return logits;
}

std::vector<std::pair<int, float>> top_k(const std::vector<float> & logits, int k) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);

    std::partial_sort(
            ids.begin(),
            ids.begin() + std::min<int>(k, ids.size()),
            ids.end(),
            [&](int a, int b) {
                return logits[a] > logits[b];
            });

    std::vector<std::pair<int, float>> out;
    for (int i = 0; i < std::min<int>(k, ids.size()); ++i) {
        out.emplace_back(ids[i], logits[ids[i]]);
    }
    return out;
}

void print_forward_stats(const std::vector<double> & times_ms) {
    if (times_ms.empty()) {
        return;
    }
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / static_cast<double>(times_ms.size());
    const double median = sorted[sorted.size() / 2];
    std::cout << "forward_ms_mean=" << mean
              << "\tforward_ms_median=" << median
              << "\tforward_ms_min=" << sorted.front()
              << "\tforward_ms_max=" << sorted.back()
              << "\trepeats=" << times_ms.size()
              << "\n";
}

args parse_args(int argc, char ** argv) {
    args out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            out.model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            out.image_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            out.mode = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            out.threads = std::stoi(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            out.top_k = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            out.warmup = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            out.repeat = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            out.benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: run_resnet50_conv_svd --model <model.gguf> --image <image> [--mode fold|im2col] [--threads N] [--top-k K] [--benchmark --warmup N --repeat N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }

    if (out.model_path.empty() || out.image_path.empty()) {
        throw std::runtime_error("both --model and --image are required");
    }
    if (out.mode != "fold" && out.mode != "im2col") {
        throw std::runtime_error("--mode must be either fold or im2col");
    }
    if (out.warmup < 0 || out.repeat <= 0) {
        throw std::runtime_error("--warmup must be >= 0 and --repeat must be > 0");
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        ggml_cpu_init();

        const args cli = parse_args(argc, argv);
#ifdef RESNET50_USE_ONEDNN
        omp_set_dynamic(0);
        omp_set_num_threads(cli.threads);
#endif
        const resnet50_model model = load_model(cli.model_path);
        const image_u8 image = load_image_rgb(cli.image_path);
        const std::vector<float> input = preprocess_image(image, model.preproc);
        std::vector<float> logits;
        if (cli.benchmark) {
            for (int i = 0; i < cli.warmup; ++i) {
                logits = run_inference(model, input, cli.threads, cli.mode);
            }
            std::vector<double> times_ms;
            times_ms.reserve(cli.repeat);
            for (int i = 0; i < cli.repeat; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                logits = run_inference(model, input, cli.threads, cli.mode);
                const auto t1 = std::chrono::steady_clock::now();
                times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            print_forward_stats(times_ms);
        } else {
            logits = run_inference(model, input, cli.threads, cli.mode);
        }
        const auto best = top_k(logits, cli.top_k);

        for (size_t i = 0; i < best.size(); ++i) {
            const int idx = best[i].first;
            const float score = best[i].second;
            std::string label = idx < static_cast<int>(model.labels.size()) ? model.labels[idx] : "<unknown>";
            std::cout << i + 1 << "\tclass_id=" << idx << "\tlogit=" << score << "\tlabel=" << label << "\n";
        }
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
