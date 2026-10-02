#include <evk_ai.h>

void benchmark_matmul() {
    printf("benchmark_matmul()\n");
    const uint32_t SIZE = 4096u;
    for (uint32_t tile_m = 48; tile_m <= 224; tile_m += 16) {
        for (uint32_t tile_n = 48; tile_n <= 224; tile_n += 16) {
            if(tile_m != 96 || tile_n != 96) {
                continue;
            }
            const uint32_t M = ((SIZE + tile_m - 1) / tile_m) * tile_m;
            const uint32_t N = ((SIZE + tile_n - 1) / tile_n) * tile_n;
            if(tile_m * tile_n >= 80*144){
                continue;
            }
            Tensor* a = new Tensor({M, SIZE});
            Tensor* b = new Tensor({SIZE, N});
            Tensor* c = new Tensor({M, N});
        
            a->identity(2.0f);
            b->identity(3.0f);

            evk::ai::GetCmd();
            for (uint32_t i = 0; i < 4; ++i) {
                evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
            }
            evk::ai::SubmitCmd(true);

            float min_ms = 1e9f;
            for(int it = 0; it < 16; ++it) {
                const uint32_t subIter = 32;
                auto& cmd = evk::ai::GetCmd();
                for (uint32_t i = 0; i < subIter; ++i) {
                    cmd.timestamp("matmul", [&]() {
                        evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
                    });
                }
                evk::ai::SubmitCmd(true);

                for(const auto& ts: evk::CmdTimestamps()) {
                    min_ms = fminf(min_ms, float(ts.end - ts.start));
                }
            }
            float tflops = float(2 * uint64_t(M) * uint64_t(N) * uint64_t(M)) / (min_ms / 1000.0f) / 1e12f;
            printf("matmul: %5.3fms (%7.3ftflops)", min_ms, tflops);
            printf(" M = %d, N = %d, tile_m = %d, tile_n = %d\n", M, N, tile_m, tile_n);

            delete a;
            delete b;
            delete c;
        }
    }
}

void benchmark_matmul_broadcast() {
    printf("benchmark_matmul_broadcast()\n");
    const uint32_t BATCH = 32u;
    const uint32_t SIZE = 2048u;
    const uint32_t TILE = 64u;
    Tensor* a = new Tensor({BATCH, SIZE, SIZE});
    Tensor* b = new Tensor({SIZE, SIZE});
    Tensor* c = new Tensor({BATCH, SIZE, SIZE});

    a->identity(2.0f);
    b->identity(3.0f);

    evk::ai::GetCmd();
    for (uint32_t i = 0; i < 4; ++i) {
        evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
    }
    evk::ai::SubmitCmd(true);

    float min_ms = 1e9f;
    for(int it = 0; it < 16; ++it) {
        const uint32_t subIter = 16;
        auto& cmd = evk::ai::GetCmd();
        for (uint32_t i = 0; i < subIter; ++i) {
            cmd.timestamp("matmul_broadcast", [&]() {
                evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
            });
        }
        
        evk::ai::SubmitCmd(true);
        for(const auto& ts: evk::CmdTimestamps()) {
            min_ms = fminf(min_ms, float(ts.end - ts.start));
        }
    }
    float tflops = float(2 * uint64_t(BATCH) * uint64_t(SIZE) * uint64_t(SIZE) * uint64_t(SIZE)) / (min_ms / 1000.0f) / 1e12f;
    printf("matmul_batched: %5.3fms (%7.3ftflops)", min_ms, tflops);
    printf(" BATCH = %d, SIZE = %d\n", BATCH, SIZE);

    delete a;
    delete b;
    delete c;
}

void benchmark_cross_entropy_loss() {
    printf("benchmark_cross_entropy_loss()\n");
    const uint32_t BATCH = 8u;
    const uint32_t SEQ = 1024u;
    const uint32_t VOCAB = 4096u*4u;
    const uint32_t POSITIONS = BATCH * SEQ;

    Tensor* logits = new Tensor({POSITIONS, VOCAB});
    Tensor* targets = new Tensor({POSITIONS});
    Tensor* grad = new Tensor({POSITIONS, VOCAB});
    Tensor* loss = new Tensor({1});

    logits->random();

    float16_t* target_data = targets->cpu();
    for (uint32_t i = 0; i < POSITIONS; ++i) {
        uint16_t idx = uint16_t((i % (VOCAB - 1)) + 1);
        if ((i % 13u) == 0u) {
            idx = 0;
        }
        target_data[i].value = idx;
    }
    targets->cpu_upload();

    evk::ai::GetCmd();
    for (uint32_t i = 0; i < 4; ++i) {
        evk::ai::cross_entropy_loss(*logits, *targets, *grad, *loss);
    }
    evk::ai::SubmitCmd(true);

    float min_ms = 1e9f;
    for (int it = 0; it < 16; ++it) {
        const uint32_t subIter = 16;
        auto& cmd = evk::ai::GetCmd();
        for (uint32_t i = 0; i < subIter; ++i) {
            cmd.timestamp("cross_entropy_loss", [&]() {
                evk::ai::cross_entropy_loss(*logits, *targets, *grad, *loss);
            });
        }

        evk::ai::SubmitCmd(true);
        for(const auto& ts: evk::CmdTimestamps()) {
            min_ms = fminf(min_ms, float(ts.end - ts.start));
        }
    }

    float positions_per_s = float(POSITIONS) / (min_ms / 1000.0f);
    float logits_per_s = float(POSITIONS) * float(VOCAB) / (min_ms / 1000.0f);
    printf("cross_entropy_loss: %5.3fms (%7.3f Mpos/s, %7.3f Glogits/s)",
           min_ms, positions_per_s / 1e6f, logits_per_s / 1e9f);
    printf(" BATCH = %d, SEQ = %d, VOCAB = %d\n", BATCH, SEQ, VOCAB);

    delete logits;
    delete targets;
    delete grad;
    delete loss;
}

void benchmark_graph_matmul_broadcast() {
    printf("benchmark_graph_matmul_broadcast_graph()\n");

    const uint32_t BATCH = 4u;
    const uint32_t SIZE = 1024u;
    const uint32_t TILE = 64u;
    const uint32_t WARMUP = 2u;
    const uint32_t ITERS = 8u;
    const uint32_t SUB_ITER = 4u;

    Graph graph;

    Tensor& a = graph.tensor({BATCH, SIZE, SIZE}).fill(float16_t(1.0f));
    Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(float16_t(1.0f));
    Tensor& target = graph.tensor({BATCH, SIZE, SIZE}).fill(float16_t(0.0f));
    Tensor& out = graph.matmul(a, w, TILE, TILE);
    Tensor& loss = graph.mse_loss(out, target);
    (void)loss;

    evk::ai::GetCmd();
    for (uint32_t i = 0; i < WARMUP; ++i) {
        graph.eval(true, false, false);
    }
    evk::ai::SubmitCmd(true);

    float min_ms = 1e9f;
    for (uint32_t it = 0; it < ITERS; ++it) {
        auto& cmd = evk::ai::GetCmd();
        for (uint32_t i = 0; i < SUB_ITER; ++i) {
            cmd.timestamp("graph_matmul_broadcast", [&]() {
                graph.eval(true, false, false);
            });
        }
        evk::ai::SubmitCmd(true);
        for (const auto& ts : evk::CmdTimestamps()) {
            if (strcmp(ts.name, "graph_matmul_broadcast") == 0) {
                min_ms = fminf(min_ms, float(ts.end - ts.start));
            }
        }
    }

    double ops = 6.0 * double(BATCH) * double(SIZE) * double(SIZE) * double(SIZE);
    double tflops = ops / (double(min_ms) / 1000.0) / 1e12;
    printf("graph_matmul_broadcast (fwd+bwd): %5.3fms (%7.3ftflops)", min_ms, tflops);
    printf(" BATCH = %u, SIZE = %u, TILE = %u\n", BATCH, SIZE, TILE);
}

// Median GPU duration; uploads, pipeline creation and queue waits are excluded.
template<class Fn>
double attention_gpu_ms(Fn operation, std::function<void()> prepare = {}) {
    for (int i = 0; i < 3; ++i) {
        if (prepare) prepare();
        operation(); evk::ai::SubmitCmd(true);
    }
    std::vector<double> samples;
    for (int i = 0; i < 9; ++i) {
        if (prepare) prepare();
        evk::ai::GetCmd().timestamp("attention", operation);
        evk::ai::SubmitCmd(true);
        const auto& timing = evk::CmdTimestamps().back();
        samples.push_back(timing.end - timing.start);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void benchmark_attention(uint32_t max_context = 16384u) {
    printf("attention_kernel,implementation,batch,length,width,forward_ms,backward_ms,combined_ms,workspace_bytes\n");
    // Keep both graphs resident and alternate sample order to reduce clock/order bias.
    // This increases benchmark peak allocation, not either kernel's reported workspace.
    for (uint32_t d : {64u, 256u})
        for (uint32_t n : {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
            if (n > max_context) continue;
            Graph graphs[2];
            Tensor* tensors[2][4];
            for (uint32_t impl = 0u; impl < 2u; ++impl) {
                Graph& graph = graphs[impl];
                Tensor& q = graph.tensor({1u, n, d});
                Tensor& k = graph.tensor(q.shape);
                Tensor& v = graph.tensor(q.shape);
                Tensor& out = graph.causal_attention(q, k, v, 0.0f, 0u,
                    impl ? AttentionKernel::Tiled : AttentionKernel::Materialized);
                tensors[impl][0] = &q; tensors[impl][1] = &k;
                tensors[impl][2] = &v; tensors[impl][3] = &out;
                for (Tensor* tensor : {&q, &k, &v, &out.grad()}) {
                    for (uint32_t i = 0; i < tensor->shape.count(); ++i)
                        tensor->cpu()[i] = float16_t(std::sin(float(i * 13u + (tensor == &k ? 7u : 0u))) * 0.1f);
                    tensor->cpu_upload();
                }
            }
            double medians[2][3];
            for (uint32_t pass = 0u; pass < 3u; ++pass) {
                std::vector<double> samples[2];
                for (int sample = -3; sample < 31; ++sample)
                    for (uint32_t order = 0u; order < 2u; ++order) {
                        uint32_t impl = order ^ (uint32_t(sample + 3) & 1u);
                        Tensor& out = *tensors[impl][3];
                        if (pass != 0u)
                            for (const auto& tensor : graphs[impl].nodes)
                                if (tensor.get() != &out) tensor->grad().fill(0.0f);
                        auto operation = [&] {
                            if (pass != 1u) out.forward_fn();
                            if (pass != 0u) out.backward_fn();
                        };
                        if (sample < 0) operation();
                        else evk::ai::GetCmd().timestamp("attention", operation);
                        evk::ai::SubmitCmd(true);
                        if (sample >= 0) {
                            const auto& timing = evk::CmdTimestamps().back();
                            double ms = timing.end - timing.start;
                            samples[impl].push_back(ms);
                            printf("attention_sample,%s,%u,%u,%u,%d,%.6f\n",
                                impl ? "tiled" : "legacy", n, d, pass, sample, ms);
                        }
                    }
                for (uint32_t impl = 0u; impl < 2u; ++impl) {
                    std::sort(samples[impl].begin(), samples[impl].end());
                    medians[impl][pass] = samples[impl][samples[impl].size() / 2u];
                }
            }
            for (uint32_t impl = 0u; impl < 2u; ++impl) {
                Graph& graph = graphs[impl];
                uint64_t bytes = 0;
                for (const auto& workspace : graph.workspaces)
                    bytes += uint64_t(workspace->shape.count()) * sizeof(float16_t);
                for (const auto& scratch : graph.scratch)
                    if (scratch.tensor) bytes += uint64_t(scratch.capacity) * sizeof(float16_t);
                for (const auto& tensor : graph.nodes)
                    if (std::find(std::begin(tensors[impl]), std::end(tensors[impl]), tensor.get()) == std::end(tensors[impl]))
                        bytes += uint64_t(tensor->shape.count()) * sizeof(float16_t) * (tensor->grad_tensor ? 2u : 1u);
                printf("attention_kernel,%s,1,%u,%u,%.6f,%.6f,%.6f,%llu\n",
                    impl ? "tiled" : "legacy", n, d, medians[impl][0], medians[impl][1], medians[impl][2],
                    static_cast<unsigned long long>(bytes));
            }
            fflush(stdout);
        }
    printf("attention_decode,length,width,decode_ms,cache_bytes\n");
    for (uint32_t d : {64u, 256u})
        for (uint32_t n : {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
            Tensor q({1u, 1u, d}), k({1u, n, d}), v({1u, n, d}), out(q.shape);
            q.fill(0.1f); k.fill(0.2f); v.fill(0.3f);
            double ms = attention_gpu_ms([&] { evk::ai::cached_attention(q, k, v, out, n - 1u); });
            printf("attention_decode,%u,%u,%.6f,%llu\n", n, d, ms,
                static_cast<unsigned long long>(uint64_t(n) * d * 4u));
        }
}

void bench(bool attention = false, uint32_t max_context = 16384u) {
    if (attention) { benchmark_attention(max_context); return; }
    // benchmark_matmul();
    // benchmark_matmul_broadcast();
    benchmark_cross_entropy_loss();
}
