#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "layer_coop_net.h"
#include "layer_threadpool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static llama_token greedy_sample(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return (llama_token) best;
}

struct LayerCoopClientProfile {
    int64_t requests = 0;
    size_t send_bytes = 0;
    size_t recv_bytes = 0;
    double send_ms = 0.0;
    double recv_header_ms = 0.0;
    double roundtrip_ms = 0.0;
};

static bool env_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::atoi(value) != 0;
}

static std::thread start_keep_hot_thread(std::atomic<bool> & active, std::atomic<bool> & stop) {
    return std::thread([&active, &stop]() {
        volatile uint64_t x = 0x9e3779b97f4a7c15ULL;
        while (!stop.load(std::memory_order_relaxed)) {
            if (active.load(std::memory_order_relaxed)) {
                for (int i = 0; i < 200000; ++i) {
                    x ^= x << 7;
                    x ^= x >> 9;
                    x += 0x9e3779b97f4a7c15ULL;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    });
}

static llama_token send_hidden(int fd, const std::vector<int32_t> & pos, const float * hidden,
        int32_t n_embd, int32_t n_vocab, bool reset_kv, double & server_ms, LayerCoopClientProfile & profile,
        std::atomic<bool> * keep_hot_active = nullptr) {
    LayerCoopRequest req {
        LAYER_COOP_MAGIC,
        LAYER_COOP_VERSION,
        (int32_t) pos.size(),
        n_embd,
        n_vocab,
        reset_kv ? 1 : 0,
    };
    const size_t hidden_count = (size_t) req.n_tokens * n_embd;
    const size_t send_bytes = sizeof(req) + pos.size() * sizeof(int32_t) + hidden_count * sizeof(float);
    const size_t recv_bytes = sizeof(LayerCoopResponse);

    const auto t_req0 = std::chrono::steady_clock::now();
    if (!layer_coop_send_all(fd, &req, sizeof(req)) ||
            !layer_coop_send_all(fd, pos.data(), pos.size() * sizeof(int32_t)) ||
            !layer_coop_send_all(fd, hidden, hidden_count * sizeof(float))) {
        throw std::runtime_error("failed to send layer-coop request");
    }
    const auto t_send_done = std::chrono::steady_clock::now();
    LayerCoopResponse resp {};
    if (keep_hot_active) {
        keep_hot_active->store(true, std::memory_order_relaxed);
    }
    if (!layer_coop_recv_all(fd, &resp, sizeof(resp)) ||
            resp.magic != LAYER_COOP_MAGIC ||
            resp.version != LAYER_COOP_VERSION ||
            resp.status != 0 ||
            resp.n_logits != 0 ||
            resp.selected_token < 0 ||
            resp.selected_token >= n_vocab) {
        if (keep_hot_active) {
            keep_hot_active->store(false, std::memory_order_relaxed);
        }
        throw std::runtime_error("bad layer-coop response");
    }
    if (keep_hot_active) {
        keep_hot_active->store(false, std::memory_order_relaxed);
    }
    const auto t_header_done = std::chrono::steady_clock::now();

    server_ms += resp.server_decode_ms;
    profile.requests++;
    profile.send_bytes += send_bytes;
    profile.recv_bytes += recv_bytes;
    profile.send_ms += std::chrono::duration<double, std::milli>(t_send_done - t_req0).count();
    profile.recv_header_ms += std::chrono::duration<double, std::milli>(t_header_done - t_send_done).count();
    profile.roundtrip_ms += std::chrono::duration<double, std::milli>(t_header_done - t_req0).count();
    return (llama_token) resp.selected_token;
}

static int accept_tail_connection(uint16_t port) {
    if (!layer_coop_net_init()) {
        throw std::runtime_error("network init failed");
    }
    const int listen_fd = (int) socket(AF_INET, SOCK_STREAM, 0);
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

    std::cout << "waiting for layer tail server on 0.0.0.0:" << port << std::endl;
    sockaddr_in peer {};
#ifdef _WIN32
    int peer_len = sizeof(peer);
    const int fd = (int) accept((SOCKET) listen_fd, (sockaddr *) &peer, &peer_len);
#else
    socklen_t peer_len = sizeof(peer);
    const int fd = accept(listen_fd, (sockaddr *) &peer, &peer_len);
#endif
    layer_coop_close(listen_fd);
    if (fd < 0) {
        throw std::runtime_error("accept failed");
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
    char peer_ip[64] = {};
    inet_ntop(AF_INET, &peer.sin_addr, peer_ip, sizeof(peer_ip));
    std::cout << "accepted layer tail server from " << peer_ip << ":" << ntohs(peer.sin_port) << std::endl;
    return fd;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model> <tokens> <threads> <host:port> <split_m_layers> [verbose]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int32_t max_new_tokens = std::stoi(argv[2]);
    const int32_t threads = std::stoi(argv[3]);
    const std::string endpoint = argv[4];
    const bool listen_endpoint = endpoint.rfind("listen:", 0) == 0;
    std::string host;
    uint16_t port = 0;
    if (!layer_coop_parse_host_port(endpoint, host, port)) {
        std::cerr << "bad endpoint\n";
        return 1;
    }
    if (listen_endpoint) {
        host = "listen";
    }
    const int32_t split_m = argc > 5 ? std::stoi(argv[5]) : 14;
    const bool verbose = argc > 6 ? std::stoi(argv[6]) != 0 : false;

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
#ifdef _WIN32
    mp.use_mmap = false;
#endif
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) {
        std::cerr << "failed to load model\n";
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int32_t n_embd = llama_model_n_embd(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params full_cp = llama_context_default_params();
    full_cp.n_ctx = 2048;
    full_cp.n_batch = 2048;
    full_cp.n_threads = threads;
    full_cp.n_threads_batch = threads;

    std::unique_ptr<llama_context, decltype(&llama_free)> full_ctx(
        llama_init_from_model(model.get(), full_cp), llama_free);
    if (!full_ctx) {
        std::cerr << "failed to create full prefill context\n";
        return 1;
    }
    LayerThreadpoolPtr full_threadpool = layer_attach_threadpool(full_ctx.get(), threads, "layer-coop-client-full");

    llama_context_params prefix_cp = llama_context_default_params();
    prefix_cp.n_ctx = 2048;
    prefix_cp.n_batch = 2048;
    prefix_cp.n_threads = threads;
    prefix_cp.n_threads_batch = threads;
    prefix_cp.embeddings = true;
    prefix_cp.layer_split_start = 0;
    prefix_cp.layer_split_end = split_m;

    std::unique_ptr<llama_context, decltype(&llama_free)> prefix_ctx(
        llama_init_from_model(model.get(), prefix_cp), llama_free);
    if (!prefix_ctx) {
        std::cerr << "failed to create prefix context\n";
        return 1;
    }
    LayerThreadpoolPtr prefix_threadpool = layer_attach_threadpool(prefix_ctx.get(), threads, "layer-coop-client-prefix");

    const std::string prompt = "Once upon a time";
    std::vector<llama_token> tokens(prompt.size() + 8);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens <= 0) {
        std::cerr << "tokenize failed\n";
        return 1;
    }
    tokens.resize(n_tokens);

    double server_ms = 0.0;
    double pc_prefill_ms = 0.0;
    double prefix_decode_ms = 0.0;
    LayerCoopClientProfile coop_profile;
    std::atomic<bool> keep_hot_active(false);
    std::atomic<bool> keep_hot_stop(false);
    std::thread keep_hot_thread;
    if (env_enabled("LAYER_COOP_CLIENT_KEEP_HOT", false)) {
        keep_hot_thread = start_keep_hot_thread(keep_hot_active, keep_hot_stop);
        std::cerr << "[layer-coop-client] keep-hot enabled while waiting for server response\n";
    }
    auto stop_keep_hot = [&]() {
        keep_hot_active.store(false, std::memory_order_relaxed);
        keep_hot_stop.store(true, std::memory_order_relaxed);
        if (keep_hot_thread.joinable()) {
            keep_hot_thread.join();
        }
    };
    const auto t_all0 = std::chrono::steady_clock::now();

    llama_kv_cache_clear(full_ctx.get());
    llama_batch full_batch = llama_batch_init(n_tokens, 0, 1);
    full_batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        full_batch.token[i] = tokens[i];
        full_batch.pos[i] = i;
        full_batch.n_seq_id[i] = 1;
        full_batch.seq_id[i][0] = 0;
        full_batch.logits[i] = (i == n_tokens - 1);
    }

    auto t0 = std::chrono::steady_clock::now();
    int ret = llama_decode(full_ctx.get(), full_batch);
    auto t1 = std::chrono::steady_clock::now();
    llama_batch_free(full_batch);
    if (ret != 0) {
        std::cerr << "full prefill failed\n";
        return 1;
    }
    pc_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    const float * full_logits = llama_get_logits_ith(full_ctx.get(), -1);
    if (!full_logits) {
        std::cerr << "full prefill logits missing\n";
        return 1;
    }
    llama_token next_token = greedy_sample(full_logits, n_vocab);
    full_ctx.reset();
    full_threadpool.reset();

    llama_kv_cache_clear(prefix_ctx.get());
    llama_batch prefix_batch = llama_batch_init(n_tokens, 0, 1);
    prefix_batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        prefix_batch.token[i] = tokens[i];
        prefix_batch.pos[i] = i;
        prefix_batch.n_seq_id[i] = 1;
        prefix_batch.seq_id[i][0] = 0;
        prefix_batch.logits[i] = 0;
    }
    t0 = std::chrono::steady_clock::now();
    ret = llama_decode(prefix_ctx.get(), prefix_batch);
    t1 = std::chrono::steady_clock::now();
    llama_batch_free(prefix_batch);
    if (ret != 0) {
        std::cerr << "PC prefix prefill failed\n";
        return 1;
    }
    pc_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int fd = listen_endpoint ? accept_tail_connection(port) : layer_coop_connect(host, port);
    std::cout << "layer cooperative decode enabled: pc_prefill=full-model prompt-only"
              << " prefix_layers=[0," << split_m
              << ") tail_endpoint=" << (listen_endpoint ? std::string("listen") : host) << ":" << port
              << " n_embd=" << n_embd << " n_vocab=" << n_vocab << std::endl;

    std::vector<llama_token> generated;
    generated.reserve(max_new_tokens);
    int32_t total_tokens = n_tokens;
    const auto t_steady0 = std::chrono::steady_clock::now();
    for (int32_t gen = 0; gen < max_new_tokens; ++gen) {
        const llama_token next = next_token;
        generated.push_back(next);
        if (verbose) {
            char buf[256];
            const int32_t len = llama_token_to_piece(vocab, next, buf, (int32_t) sizeof(buf), 0, true);
            std::cout << "token #" << gen << " id=" << next
                      << " piece=\"" << (len > 0 ? std::string(buf, buf + len) : "[ERR]") << "\"\n";
        }

        llama_batch gb = llama_batch_init(1, 0, 1);
        gb.n_tokens = 1;
        gb.token[0] = next;
        gb.pos[0] = total_tokens;
        gb.n_seq_id[0] = 1;
        gb.seq_id[0][0] = 0;
        gb.logits[0] = 1;
        t0 = std::chrono::steady_clock::now();
        ret = llama_decode(prefix_ctx.get(), gb);
        t1 = std::chrono::steady_clock::now();
        llama_batch_free(gb);
        if (ret != 0) {
            std::cerr << "prefix decode failed\n";
            return 1;
        }
        prefix_decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float * hidden = llama_get_embeddings_ith(prefix_ctx.get(), 0);
        if (!hidden) {
            std::cerr << "prefix decode embeddings missing\n";
            return 1;
        }
        int32_t p = total_tokens;
        next_token = send_hidden(fd, std::vector<int32_t>{p}, hidden, n_embd, n_vocab, gen == 0, server_ms, coop_profile,
                keep_hot_thread.joinable() ? &keep_hot_active : nullptr);
        total_tokens++;
    }
    const auto t_steady1 = std::chrono::steady_clock::now();
    const auto t_all1 = std::chrono::steady_clock::now();
    layer_coop_close(fd);
    stop_keep_hot();

    std::string text;
    for (llama_token tok : generated) {
        char buf[256];
        const int32_t len = llama_token_to_piece(vocab, tok, buf, (int32_t) sizeof(buf), 0, true);
        text += len > 0 ? std::string(buf, buf + len) : "[ERR]";
    }
    const double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
    const double steady_decode_ms = std::chrono::duration<double, std::milli>(t_steady1 - t_steady0).count();
    const double generated_count = (double) generated.size();
    std::cout << "Generated text: " << text << std::endl;
    std::cout << "[layer-coop-client] generated=" << generated.size()
              << " pc_prefill_ms=" << pc_prefill_ms
              << " prefix_decode_ms=" << prefix_decode_ms
              << " server_decode_ms=" << server_ms
              << " total_ms=" << total_ms
              << " throughput=" << (generated.empty() ? 0.0 : generated.size() / (total_ms / 1000.0))
              << " tok/s" << std::endl;
    std::cout << "[layer-coop-steady] generated=" << generated.size()
              << " steady_decode_ms=" << steady_decode_ms
              << " steady_throughput=" << (generated.empty() ? 0.0 : generated_count / (steady_decode_ms / 1000.0))
              << " tok/s"
              << " steady_ms_per_token=" << (generated.empty() ? 0.0 : steady_decode_ms / generated_count)
              << " prefix_decode_ms=" << prefix_decode_ms
              << " server_decode_ms=" << server_ms
              << " network_wait_ms=" << (coop_profile.roundtrip_ms - server_ms)
              << std::endl;
    std::cout << "[layer-coop-profile] requests=" << coop_profile.requests
              << " send_bytes=" << coop_profile.send_bytes
              << " recv_bytes=" << coop_profile.recv_bytes
              << " send_ms=" << coop_profile.send_ms
              << " recv_header_ms=" << coop_profile.recv_header_ms
              << " roundtrip_ms=" << coop_profile.roundtrip_ms
              << " client_wait_other_ms=" << (coop_profile.roundtrip_ms - coop_profile.send_ms - coop_profile.recv_header_ms)
              << std::endl;
    std::cout << "SUCCEED (layer cooperative decode done)" << std::endl;
    return 0;
}
