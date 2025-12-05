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

            evk::ai::BeginGraphRecording();
            for (uint32_t i = 0; i < 4; ++i) {
                evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
            }
            evk::ai::EndGraphRecording(true);

            float min_ms = 1e9f;
            for(int it = 0; it < 16; ++it) {
                const uint32_t subIter = 32;
                evk::ai::BeginGraphRecording();
                auto& cmd = evk::ai::GetCmd();
                for (uint32_t i = 0; i < subIter; ++i) {
                    cmd.timestamp("matmul", [&]() {
                        evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
                    });
                }
                evk::ai::EndGraphRecording(true);

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

    evk::ai::BeginGraphRecording();
    for (uint32_t i = 0; i < 4; ++i) {
        evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
    }
    evk::ai::EndGraphRecording(true);

    float min_ms = 1e9f;
    for(int it = 0; it < 16; ++it) {
        const uint32_t subIter = 16;
        evk::ai::BeginGraphRecording();
        auto& cmd = evk::ai::GetCmd();
        for (uint32_t i = 0; i < subIter; ++i) {
            cmd.timestamp("matmul_broadcast", [&]() {
                evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
            });
        }
        
        evk::ai::EndGraphRecording(true);
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

void bench() {
    benchmark_matmul();
    // benchmark_matmul_broadcast();
}