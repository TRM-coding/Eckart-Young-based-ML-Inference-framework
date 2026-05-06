#define RESNET50_NO_MAIN
#include "run_resnet50.cpp"
#include "resnet_coop_net.h"

#include <chrono>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <threads> <host:port> <expected_split_blocks>\n";
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        const int threads = std::stoi(argv[2]);
        std::string host;
        uint16_t port = 0;
        if (!layer_coop_parse_host_port(argv[3], host, port)) {
            throw std::runtime_error("bad host:port");
        }
        const int expected_split = std::stoi(argv[4]);
        const resnet50_model model = load_model(model_path);
        const int fd = layer_coop_connect(host, port);
        while (true) {
            ResnetCoopRequest req {};
            if (!layer_coop_recv_all(fd, &req, sizeof(req))) {
                break;
            }
            if (req.magic != RESNET_COOP_MAGIC ||
                    req.version != RESNET_COOP_VERSION ||
                    req.split_blocks != expected_split ||
                    req.width <= 0 ||
                    req.height <= 0 ||
                    req.channels <= 0) {
                throw std::runtime_error("bad request");
            }
            feature_map feat;
            feat.width = req.width;
            feat.height = req.height;
            feat.channels = req.channels;
            feat.data.resize(static_cast<size_t>(feat.width) * feat.height * feat.channels);
            if (!layer_coop_recv_all(fd, feat.data.data(), feat.data.size() * sizeof(float))) {
                break;
            }
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> logits = run_tail_logits(model, std::move(feat), req.split_blocks, threads);
            auto t1 = std::chrono::steady_clock::now();
            const auto best = top_k(logits, 1);
            ResnetCoopResponse resp {
                RESNET_COOP_MAGIC,
                RESNET_COOP_VERSION,
                0,
                best.empty() ? -1 : best[0].first,
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
            };
            if (!layer_coop_send_all(fd, &resp, sizeof(resp))) {
                break;
            }
        }
        layer_coop_close(fd);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
