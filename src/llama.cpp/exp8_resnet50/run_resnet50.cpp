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
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef RESNET50_USE_ONEDNN
#include <omp.h>
#endif
#ifdef RESNET50_USE_ONEDNN
#include "oneapi/dnnl/dnnl.hpp"
#endif

namespace {

#if !defined(__ANDROID__)
extern "C" void ggml_svd_local_profile_print_and_reset(void) {}
#endif

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
    int threads = 8;
    int top_k = 5;
    int warmup = 1;
    int repeat = 1;
    bool benchmark = false;
};

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
    const ggml_tensor * weight = nullptr;
    const ggml_tensor * bias = nullptr;
    std::string src_blob;
    int stride = 0;
    int padding = 0;

    bool operator==(const onednn_conv_key & other) const {
        return weight == other.weight &&
               bias == other.bias &&
               src_blob == other.src_blob &&
               stride == other.stride &&
               padding == other.padding;
    }
};

struct onednn_conv_key_hash {
    size_t operator()(const onednn_conv_key & key) const {
        size_t h = std::hash<const void *>{}(key.weight);
        h ^= std::hash<const void *>{}(key.bias) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(key.src_blob) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
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

const onednn_conv_plan & get_onednn_conv_plan(
        const ggml_tensor * weight,
        const ggml_tensor * bias,
        const dnnl::memory::desc & src_md,
        int input_w,
        int input_h,
        int input_c,
        int stride,
        int padding) {
    static std::unordered_map<onednn_conv_key, onednn_conv_plan, onednn_conv_key_hash> cache;

    onednn_conv_key key{weight, bias, blob_key(src_md), stride, padding};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    dnnl::engine & eng = onednn_engine();

    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    const int out_w = (input_w + 2 * padding - kw) / stride + 1;
    const int out_h = (input_h + 2 * padding - kh) / stride + 1;
    const dnnl::memory::dims weight_dims = {out_channels, input_c, kh, kw};
    const dnnl::memory::dims bias_dims = {out_channels};
    const dnnl::memory::dims strides = {stride, stride};
    const dnnl::memory::dims pads = {padding, padding};

    const auto user_weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::oihw);
    const auto bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::x);
    const auto dst_md = dnnl::memory::desc({1, out_channels, out_h, out_w}, dt::f32, tag::any);

    const auto conv_pd = dnnl::convolution_forward::primitive_desc(
            eng,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            src_md,
            dnnl::memory::desc(weight_dims, dt::f32, tag::any),
            bias_md,
            dst_md,
            strides,
            pads,
            pads);

    dnnl::memory user_weight_memory(user_weight_md, eng, weight->data);
    dnnl::memory weight_memory(conv_pd.weights_desc(), eng);
    if (conv_pd.weights_desc() != user_weight_md) {
        dnnl::reorder(user_weight_memory, weight_memory).execute(onednn_stream(), user_weight_memory, weight_memory);
        onednn_stream().wait();
    } else {
        weight_memory = user_weight_memory;
    }

    dnnl::memory bias_memory(bias_md, eng, bias ? bias->data : nullptr);
    dnnl::convolution_forward conv(conv_pd);
    auto inserted = cache.emplace(
            key,
            onednn_conv_plan(out_w, out_h, out_channels, conv_pd, conv, weight_memory, bias_memory));
    return inserted.first->second;
}

feature_map conv2d_onednn(
        const ggml_tensor * weight,
        const ggml_tensor * bias,
        feature_map & input,
        int stride,
        int padding) {
    const int in_channels = static_cast<int>(weight->ne[2]);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv input channel mismatch");
    }

    ensure_onednn(input);
    const dnnl::memory::desc src_desc = input.onednn_memory.get_desc();
    const onednn_conv_plan & plan = get_onednn_conv_plan(
            weight, bias, src_desc, input.width, input.height, input.channels, stride, padding);
    dnnl::stream & stream = onednn_stream();
    const dnnl::memory::desc plan_src_desc = plan.pd.src_desc();
    ensure_onednn(input, &plan_src_desc);

    feature_map out;
    out.width = plan.out_w;
    out.height = plan.out_h;
    out.channels = plan.out_c;
    bind_onednn_memory(out, plan.pd.dst_desc());
    plan.conv.execute(
            stream,
            {
                    {DNNL_ARG_SRC, input.onednn_memory},
                    {DNNL_ARG_WEIGHTS, plan.weights_memory},
                    {DNNL_ARG_BIAS, plan.bias_memory},
                    {DNNL_ARG_DST, out.onednn_memory},
            });
    stream.wait();
    out.data_valid = false;
    return out;
}

struct onednn_eltwise_plan {
    dnnl::eltwise_forward::primitive_desc pd;
    dnnl::eltwise_forward prim;

    onednn_eltwise_plan(const dnnl::eltwise_forward::primitive_desc & pd, const dnnl::eltwise_forward & prim)
        : pd(pd), prim(prim) {}
};

const onednn_eltwise_plan & get_onednn_relu_plan(const dnnl::memory::desc & src_md) {
    static std::unordered_map<onednn_unary_key, onednn_eltwise_plan, onednn_unary_key_hash> cache;
    onednn_unary_key key{blob_key(src_md), 1, 0, 0, 0};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    const auto pd = dnnl::eltwise_forward::primitive_desc(
            onednn_engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu, src_md, src_md, 0.0f, 0.0f);
    auto inserted = cache.emplace(key, onednn_eltwise_plan(pd, dnnl::eltwise_forward(pd)));
    return inserted.first->second;
}

struct onednn_pool_plan {
    int out_w = 0;
    int out_h = 0;
    dnnl::pooling_forward::primitive_desc pd;
    dnnl::pooling_forward prim;

    onednn_pool_plan(int out_w, int out_h, const dnnl::pooling_forward::primitive_desc & pd, const dnnl::pooling_forward & prim)
        : out_w(out_w), out_h(out_h), pd(pd), prim(prim) {}
};

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

struct onednn_sum_plan {
    dnnl::sum::primitive_desc pd;
    dnnl::sum prim;

    onednn_sum_plan(const dnnl::sum::primitive_desc & pd, const dnnl::sum & prim)
        : pd(pd), prim(prim) {}
};

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

void relu_onednn_inplace(feature_map & fm) {
    ensure_onednn(fm);
    const onednn_eltwise_plan & plan = get_onednn_relu_plan(fm.onednn_memory.get_desc());
    feature_map out;
    out.width = fm.width;
    out.height = fm.height;
    out.channels = fm.channels;
    bind_onednn_memory(out, fm.onednn_memory.get_desc());
    plan.prim.execute(
            onednn_stream(),
            {
                    {DNNL_ARG_SRC, fm.onednn_memory},
                    {DNNL_ARG_DST, out.onednn_memory},
            });
    onednn_stream().wait();
    out.data_valid = false;
    fm = std::move(out);
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
    plan.prim.execute(
            onednn_stream(),
            {
                    {DNNL_ARG_SRC, input.onednn_memory},
                    {DNNL_ARG_DST, out.onednn_memory},
            });
    onednn_stream().wait();
    out.data_valid = false;
    return out;
}

void add_onednn_inplace(feature_map & dst, feature_map & src) {
    ensure_onednn(dst);
    const dnnl::memory::desc dst_desc = dst.onednn_memory.get_desc();
    ensure_onednn(src, &dst_desc);
    const onednn_sum_plan & plan = get_onednn_sum_plan(dst.onednn_memory.get_desc());
    feature_map out;
    out.width = dst.width;
    out.height = dst.height;
    out.channels = dst.channels;
    bind_onednn_memory(out, dst.onednn_memory.get_desc());
    plan.prim.execute(
            onednn_stream(),
            {
                    {DNNL_ARG_MULTIPLE_SRC + 0, dst.onednn_memory},
                    {DNNL_ARG_MULTIPLE_SRC + 1, src.onednn_memory},
                    {DNNL_ARG_DST, out.onednn_memory},
            });
    onednn_stream().wait();
    out.data_valid = false;
    dst = std::move(out);
}
#endif

feature_map conv2d_ggml_op(
        const ggml_tensor * weight,
        const ggml_tensor * bias,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads) {
#ifdef RESNET50_USE_ONEDNN
    (void)n_threads;
    return conv2d_onednn(weight, bias, const_cast<feature_map &>(input), stride, padding);
#endif

    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv input channel mismatch");
    }

    const int out_w = (input.width + 2 * padding - kw) / stride + 1;
    const int out_h = (input.height + 2 * padding - kh) / stride + 1;
    const size_t bytes =
            8 * 1024 * 1024 +
            sizeof(float) * (
                    input.data.size() +
                    static_cast<size_t>(out_w * out_h * out_channels) +
                    static_cast<size_t>(out_w * out_h * kw * kh * in_channels));

    ggml_init_params params = {
        /*.mem_size   =*/ bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_ctx_ptr ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for conv_2d");
    }

    ggml_tensor * input_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, input.width, input.height, input.channels, 1);
    if (input_tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml input for conv_2d");
    }
    std::memcpy(input_tensor->data, input.data.data(), sizeof(float) * input.data.size());

    ggml_tensor * output = ggml_conv_2d(ctx.get(), const_cast<ggml_tensor *>(weight), input_tensor, stride, stride, padding, padding, 1, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    if (graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed for conv_2d");
    }

    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute_with_ctx failed for conv_2d");
    }

    feature_map out;
    out.width = out_w;
    out.height = out_h;
    out.channels = out_channels;
    out.data.assign(static_cast<size_t>(out.width * out.height * out.channels), 0.0f);

    const float * output_ptr = static_cast<const float *>(output->data);
    const int64_t out_plane = static_cast<int64_t>(out.width) * out.height;
    parallel_for_chunks(out_channels, n_threads, [&](int oc_begin, int oc_end) {
        for (int oc = oc_begin; oc < oc_end; ++oc) {
            const float b = bias ? bias_value(bias, oc) : 0.0f;
            const size_t dst_base = static_cast<size_t>(out_plane * oc);
            for (int64_t pos = 0; pos < out_plane; ++pos) {
                out.data[dst_base + static_cast<size_t>(pos)] =
                        output_ptr[static_cast<size_t>(pos + out_plane * oc)] + b;
            }
        }
    });

    return out;
}

feature_map conv2d(
        const ggml_tensor * weight,
        const ggml_tensor * bias,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads) {
    return conv2d_ggml_op(weight, bias, input, stride, padding, n_threads);
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
        int n_threads) {
    const std::string base = "resnet.stage." + std::to_string(stage_idx) + ".block." + std::to_string(block_idx);

    feature_map identity = input;
    if (const ggml_tensor * downsample = optional_tensor(model.weights.get(), base + ".downsample.weight")) {
        identity = conv2d(
                downsample,
                require_tensor(model.weights.get(), base + ".downsample.bias"),
                identity,
                block_idx == 0 && stage_idx > 0 ? 2 : 1,
                0,
                n_threads);
    }

    feature_map out = conv2d(
            require_tensor(model.weights.get(), base + ".conv1.weight"),
            require_tensor(model.weights.get(), base + ".conv1.bias"),
            input,
            1,
            0,
            n_threads);
    relu_inplace(out);

    out = conv2d(
            require_tensor(model.weights.get(), base + ".conv2.weight"),
            require_tensor(model.weights.get(), base + ".conv2.bias"),
            out,
            block_idx == 0 && stage_idx > 0 ? 2 : 1,
            1,
            n_threads);
    relu_inplace(out);

    out = conv2d(
            require_tensor(model.weights.get(), base + ".conv3.weight"),
            require_tensor(model.weights.get(), base + ".conv3.bias"),
            out,
            1,
            0,
            n_threads);

    add_inplace(out, identity);
    relu_inplace(out);
    return out;
}

std::vector<float> run_inference(const resnet50_model & model, const std::vector<float> & input, int n_threads) {
    const int image_size = static_cast<int>(model.preproc.image_size);
    feature_map cur;
    cur.width = image_size;
    cur.height = image_size;
    cur.channels = 3;
    cur.data = input;

    cur = conv2d(
            require_tensor(model.weights.get(), "resnet.stem.conv.weight"),
            require_tensor(model.weights.get(), "resnet.stem.conv.bias"),
            cur,
            2,
            3,
            n_threads);
    relu_inplace(cur);
    cur = max_pool_2d(cur, 3, 2, 1);

    for (int stage_idx = 0; stage_idx < 4; ++stage_idx) {
        for (uint32_t block_idx = 0; block_idx < model.stage_block_count[stage_idx]; ++block_idx) {
            cur = bottleneck_block(model, cur, stage_idx, static_cast<int>(block_idx), n_threads);
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

int total_bottleneck_blocks(const resnet50_model & model) {
    int total = 0;
    for (uint32_t n : model.stage_block_count) {
        total += static_cast<int>(n);
    }
    return total;
}

feature_map run_prefix_feature(
        const resnet50_model & model,
        const std::vector<float> & input,
        int split_blocks,
        int n_threads) {
    const int image_size = static_cast<int>(model.preproc.image_size);
    feature_map cur;
    cur.width = image_size;
    cur.height = image_size;
    cur.channels = 3;
    cur.data = input;

    cur = conv2d(
            require_tensor(model.weights.get(), "resnet.stem.conv.weight"),
            require_tensor(model.weights.get(), "resnet.stem.conv.bias"),
            cur,
            2,
            3,
            n_threads);
    relu_inplace(cur);
    cur = max_pool_2d(cur, 3, 2, 1);

    int block_linear = 0;
    for (int stage_idx = 0; stage_idx < 4; ++stage_idx) {
        for (uint32_t block_idx = 0; block_idx < model.stage_block_count[stage_idx]; ++block_idx) {
            if (block_linear >= split_blocks) {
                return cur;
            }
            cur = bottleneck_block(model, cur, stage_idx, static_cast<int>(block_idx), n_threads);
            ++block_linear;
        }
    }
    return cur;
}

feature_map run_tail_feature(
        const resnet50_model & model,
        feature_map cur,
        int split_blocks,
        int n_threads) {
    int block_linear = 0;
    for (int stage_idx = 0; stage_idx < 4; ++stage_idx) {
        for (uint32_t block_idx = 0; block_idx < model.stage_block_count[stage_idx]; ++block_idx) {
            if (block_linear >= split_blocks) {
                cur = bottleneck_block(model, cur, stage_idx, static_cast<int>(block_idx), n_threads);
            }
            ++block_linear;
        }
    }
    return cur;
}

std::vector<float> classify_feature(const resnet50_model & model, feature_map & cur, int n_threads) {
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

std::vector<float> run_tail_logits(
        const resnet50_model & model,
        feature_map cur,
        int split_blocks,
        int n_threads) {
    cur = run_tail_feature(model, std::move(cur), split_blocks, n_threads);
    return classify_feature(model, cur, n_threads);
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
    const double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    const double mean = sum / static_cast<double>(times_ms.size());
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
            std::cout << "usage: run_resnet50 --model <model.gguf> --image <image> [--threads N] [--top-k K] [--benchmark --warmup N --repeat N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }

    if (out.model_path.empty() || out.image_path.empty()) {
        throw std::runtime_error("both --model and --image are required");
    }
    if (out.warmup < 0 || out.repeat <= 0) {
        throw std::runtime_error("--warmup must be >= 0 and --repeat must be > 0");
    }
    return out;
}

} // namespace

#ifndef RESNET50_NO_MAIN
int main(int argc, char ** argv) {
    try {
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
                logits = run_inference(model, input, cli.threads);
            }

            std::vector<double> times_ms;
            times_ms.reserve(cli.repeat);
            for (int i = 0; i < cli.repeat; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                logits = run_inference(model, input, cli.threads);
                const auto t1 = std::chrono::steady_clock::now();
                times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            print_forward_stats(times_ms);
        } else {
            logits = run_inference(model, input, cli.threads);
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
#endif
