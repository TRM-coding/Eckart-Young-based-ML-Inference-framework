#define RESNET50_NO_MAIN
#include "run_resnet50.cpp"
#include "resnet_coop_net.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

static int accept_connection(uint16_t port) {
    if (!layer_coop_net_init()) {
        throw std::runtime_error("network init failed");
    }
    const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        throw std::runtime_error("socket failed");
    }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(listen_fd, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(listen_fd, 1) != 0) {
        layer_coop_close(listen_fd);
        throw std::runtime_error("bind/listen failed");
    }
    sockaddr_in peer {};
    socklen_t peer_len = sizeof(peer);
    const int fd = accept(listen_fd, (sockaddr *) &peer, &peer_len);
    layer_coop_close(listen_fd);
    if (fd < 0) {
        throw std::runtime_error("accept failed");
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
    return fd;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <image> <threads> <port> <split_blocks> <repeat> [warmup]\n";
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        const std::string image_path = argv[2];
        const int threads = std::stoi(argv[3]);
        const uint16_t port = static_cast<uint16_t>(std::stoi(argv[4]));
        const int split_blocks = std::stoi(argv[5]);
        const int repeat = std::stoi(argv[6]);
        const int warmup = argc > 7 ? std::stoi(argv[7]) : 0;

        const resnet50_model model = load_model(model_path);
        const image_u8 image = load_image_rgb(image_path);
        const std::vector<float> input = preprocess_image(image, model.preproc);
        const int fd = accept_connection(port);

        double prefix_ms = 0.0;
        double send_ms = 0.0;
        double wait_ms = 0.0;
        double server_ms = 0.0;
        int top1 = -1;

        const int total = warmup + repeat;
        auto all0 = std::chrono::steady_clock::now();
        for (int i = 0; i < total; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            feature_map feat = run_prefix_feature(model, input, split_blocks, threads);
#ifdef RESNET50_USE_ONEDNN
            ensure_plain(feat);
#endif
            auto t1 = std::chrono::steady_clock::now();
            ResnetCoopRequest req {
                RESNET_COOP_MAGIC,
                RESNET_COOP_VERSION,
                split_blocks,
                feat.width,
                feat.height,
                feat.channels,
            };
            const size_t payload = feat.data.size() * sizeof(float);
            if (!layer_coop_send_all(fd, &req, sizeof(req)) ||
                    !layer_coop_send_all(fd, feat.data.data(), payload)) {
                throw std::runtime_error("send failed");
            }
            auto t2 = std::chrono::steady_clock::now();
            ResnetCoopResponse resp {};
            if (!layer_coop_recv_all(fd, &resp, sizeof(resp)) ||
                    resp.magic != RESNET_COOP_MAGIC ||
                    resp.version != RESNET_COOP_VERSION ||
                    resp.status != 0) {
                throw std::runtime_error("bad response");
            }
            auto t3 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                prefix_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                send_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
                wait_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
                server_ms += resp.server_ms;
            }
            top1 = resp.top1;
        }
        auto all1 = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(all1 - all0).count();
        const double measured_ms = total_ms; // includes warmup, reported for wall-clock sanity.
        const double avg_ms = (prefix_ms + send_ms + wait_ms) / std::max(1, repeat);
        const double throughput = 1000.0 / avg_ms;
        std::cout << "[resnet-coop-steady]"
                  << " repeat=" << repeat
                  << " split_blocks=" << split_blocks
                  << " top1=" << top1
                  << " avg_ms=" << avg_ms
                  << " throughput=" << throughput
                  << " images/s"
                  << " prefix_ms=" << (prefix_ms / std::max(1, repeat))
                  << " send_ms=" << (send_ms / std::max(1, repeat))
                  << " wait_ms=" << (wait_ms / std::max(1, repeat))
                  << " server_ms=" << (server_ms / std::max(1, repeat))
                  << " wall_ms=" << measured_ms
                  << "\n";
        layer_coop_close(fd);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
