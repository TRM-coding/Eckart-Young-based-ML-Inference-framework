#pragma once

#include "../exp6_decode_svd_model/layer_coop_net.h"

#include <cstdint>

struct ResnetCoopRequest {
    uint32_t magic;
    uint32_t version;
    int32_t  split_blocks;
    int32_t  width;
    int32_t  height;
    int32_t  channels;
};

struct ResnetCoopResponse {
    uint32_t magic;
    uint32_t version;
    int32_t  status;
    int32_t  top1;
    double   server_ms;
};

constexpr uint32_t RESNET_COOP_MAGIC = 0x52434f50U; // RCOP
constexpr uint32_t RESNET_COOP_VERSION = 1;
