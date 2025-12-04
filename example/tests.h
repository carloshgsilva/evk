
#include <evk_ai.h>

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

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
        
            // warm up
            for (uint32_t i = 0; i < 4; ++i) {
                evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
            }

            float min_ms = 1e9f;
            for(int it = 0; it < 16; ++it) {
                const uint32_t subIter = 32;
                for (uint32_t i = 0; i < subIter; ++i) {
                    evk::CmdTimestamp("matmul", [&]() {
                        evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
                    });
                }
                
                evk::Sync();
                for(auto ts: evk::GetTimestamps()) {
                    min_ms = fminf(min_ms, float(ts.end - ts.start));
                }
            }
            float tflops = float(2 * uint64_t(M) * uint64_t(N) * uint64_t(M)) / (min_ms / 1000.0f) / 1e12f;
            printf("matmul: %5.3fms (%7.3ftflops)", min_ms, tflops);
            printf(" M = %d, N = %d, tile_m = %d, tile_n = %d\n", M, N, tile_m, tile_n);

            delete a;
            delete b;
            delete c;
            evk::Sync();
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

    // warm up
    for (uint32_t i = 0; i < 4; ++i) {
        evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
    }

    float min_ms = 1e9f;
    for(int it = 0; it < 16; ++it) {
        const uint32_t subIter = 16;
        for (uint32_t i = 0; i < subIter; ++i) {
            evk::CmdTimestamp("matmul_broadcast", [&]() {
                evk::ai::matmul(*a, *b, *c, false, false, false, TILE, TILE);
            });
        }
        
        evk::Sync();
        for(auto ts: evk::GetTimestamps()) {
            min_ms = fminf(min_ms, float(ts.end - ts.start));
        }
    }
    float tflops = float(2 * uint64_t(BATCH) * uint64_t(SIZE) * uint64_t(SIZE) * uint64_t(SIZE)) / (min_ms / 1000.0f) / 1e12f;
    printf("matmul_batched: %5.3fms (%7.3ftflops)", min_ms, tflops);
    printf(" BATCH = %d, SIZE = %d\n", BATCH, SIZE);

    delete a;
    delete b;
    delete c;
    evk::Sync();
}


void test_add() {
    printf("test_add()\n");
    Tensor a({4});
    Tensor b({4});
    Tensor out({4});

    float16_t* ap = a.cpu();
    float16_t* bp = b.cpu();
    for (uint32_t i = 0; i < 4; ++i) {
        ap[i] = float16_t(float(i));
        bp[i] = float16_t(float(i * 2));
    }
    a.cpu_upload();
    b.cpu_upload();

    evk::ai::add(a, b, out);

    out.cpu_download();
    float16_t* outp = out.cpu();
    for (uint32_t i = 0; i < 4; ++i) {
        TEST(outp[i] == float16_t(float(i) + float(i * 2)));
    }
}

void test_matmul() {
    printf("test_matmul()\n");

    {
        const uint32_t SIZE = 400u;
        Tensor a = Tensor({SIZE, SIZE});
        Tensor b = Tensor({SIZE, SIZE});
        Tensor c = Tensor({SIZE, SIZE});
        a.identity(2.0f);
        b.identity(2.0f);
        evk::ai::matmul(a, b, c);

        c.cpu_download();
        bool diagonal_test = true;
        bool off_diagonal_test = true;
        for(uint32_t i = 0; i < SIZE*SIZE; ++i) {
            uint32_t r = i / SIZE;
            uint32_t col = i % SIZE;
            if (r == col) {
                if(c.cpu()[i] != float16_t(4.0f)) {
                    diagonal_test = false;
                }
            } else {
                if(c.cpu()[i] != float16_t(0.0f)) {
                    off_diagonal_test = false;
                }
            }
        }
        TEST(diagonal_test);
        TEST(off_diagonal_test);
    }
}

void test_matmul_broadcast() {
    printf("test_matmul_broadcast()\n");
    const uint32_t B = 3u;
    const uint32_t M = 32u;
    const uint32_t K = 32u;
    const uint32_t N = 32u;

    Tensor a({1, M, K});
    Tensor b({K, N});
    Tensor c({B, M, N});

    a.fill(float16_t(1.0f));
    b.identity(float16_t(1.0f));

    evk::ai::matmul(a, b, c, false, false, false, 16, 16);

    c.cpu_download();
    float16_t* cp = c.cpu();
    bool ok = true;
    for (uint32_t i = 0; i < B * M * N; ++i) {
        if (cp[i] != float16_t(1.0f)) { ok = false; break; }
    }
    TEST(ok);
}

void test_mse_loss() {
    printf("test_mse_loss()\n");
    // Test 1: Simple 1D tensors
    Tensor predicted({3});
    Tensor target({3});
    Tensor mse_result({1});
    Tensor grad({3});

    // Fill with known values
    float16_t* pred_cpu = predicted.cpu();
    float16_t* target_cpu = target.cpu();

    pred_cpu[0] = float16_t(1.0f);
    pred_cpu[1] = float16_t(2.0f);
    pred_cpu[2] = float16_t(3.0f);

    target_cpu[0] = float16_t(1.0f);
    target_cpu[1] = float16_t(2.0f);
    target_cpu[2] = float16_t(4.0f);  // Different from predicted

    predicted.cpu_upload();
    target.cpu_upload();

    // Compute MSE loss
    evk::ai::mse_loss(predicted, target, grad, mse_result);

    // Download result and verify
    mse_result.cpu_download();
    float16_t* mse_cpu = mse_result.cpu();

    // Expected MSE
    float expected_mse = 1.0f / 3.0f;
    float actual_mse = float(mse_cpu[0]);
    TEST(std::abs(actual_mse - expected_mse) < 1e-4f);

    grad.cpu_download();
    float16_t* grad_cpu = grad.cpu();
    TEST(grad_cpu[0] == float16_t(0.0f));
    TEST(grad_cpu[1] == float16_t(0.0f));
    TEST(grad_cpu[2] == float16_t(1.0f));
}

void test_flash_attention_backward() {
    printf("test_flash_attention_backward()\n");

    const uint32_t B = 1u;
    const uint32_t N = 16u; // tile-aligned
    const uint32_t Dh = 16u; // tile-aligned
    const uint32_t H = 2u;
    const uint32_t D = H * Dh;

    Tensor q({B, N, D});
    Tensor k({B, N, Dh});
    Tensor v({B, N, Dh});
    Tensor o({B, N, D});
    Tensor dO({B, N, D});
    Tensor dQ({B, N, D});
    Tensor dK({B, N, Dh});
    Tensor dV({B, N, Dh});

    // Use values such that softmax uniform (K=0), V=0 -> simplifies expectations
    {
        float16_t* qp = q.cpu();
        for (uint32_t i = 0; i < q.shape.count(); ++i) qp[i] = float16_t(0.0f);
        q.cpu_upload();
        float16_t* kp = k.cpu();
        for (uint32_t i = 0; i < k.shape.count(); ++i) kp[i] = float16_t(0.0f);
        k.cpu_upload();
        float16_t* vp = v.cpu();
        for (uint32_t i = 0; i < v.shape.count(); ++i) vp[i] = float16_t(0.0f);
        v.cpu_upload();
    }

    // dO pattern: d for each channel
    {
        float16_t* dop = dO.cpu();
        for (uint32_t d = 0; d < D; ++d) dop[d] = float16_t(float(d + 1));
        dO.cpu_upload();
    }

    evk::ai::flash_attention_bwd(q, k, v, o, dO, dQ, dK, dV);

    dQ.cpu_download();
    dK.cpu_download();
    dV.cpu_download();

    // With K=0 and V=0, softmax is uniform; dQ and dK should be ~0, dV equals row-wise mean of dO per kd across i=0..N-1
    bool ok_dQ = true, ok_dK = true, ok_dV = true;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            if (std::abs(float(dQ.cpu()[i*D + d])) > 1e-2f) ok_dQ = false;
        }
    }
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t kd = 0; kd < Dh; ++kd) {
            if (std::abs(float(dK.cpu()[j*Dh + kd])) > 1e-2f) ok_dK = false;
        }
    }
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t kd = 0; kd < Dh; ++kd) {
            float mean = 0.0f;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t h = 0; h < H; ++h) mean += float(dO.cpu()[i*D + h*Dh + kd]);
            }
            mean /= float(N);
            float got = float(dV.cpu()[j*Dh + kd]);
            if (std::abs(got - mean) > 5e-2f) ok_dV = false;
        }
    }
    TEST(ok_dQ);
    TEST(ok_dK);
    TEST(ok_dV);
}

void test_flash_attention_cmp_softmax() {
    printf("test_flash_attention_cmp_softmax()\n");
    const uint32_t B = 1u;
    const uint32_t N = 1024u*32u;   // tile aligned
    const uint32_t Dh = 64u;  // tile aligned
    const uint32_t H = 1u;
    const uint32_t D = H * Dh;

    Tensor q({B, N, D});
    Tensor k({B, N, Dh});
    Tensor v({B, N, Dh});
    Tensor o_flash({B, N, D});

    Tensor q_scaled({B, N, D});
    Tensor s({B, N, N});
    Tensor p({B, N, N});
    Tensor o_base({B, N, D});

    {
        float16_t* qp = q.cpu();
        float16_t* kp = k.cpu();
        float16_t* vp = v.cpu();
        for (uint32_t i = 0; i < N; ++i) {
            for (uint32_t d = 0; d < D; ++d) {
                qp[i * D + d] = float16_t(float((i + d) % 13) / 13.0f);
            }
        }
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t kd = 0; kd < Dh; ++kd) {
                kp[j * Dh + kd] = float16_t(float((j * 7 + kd) % 11) / 11.0f);
                vp[j * Dh + kd] = float16_t(float((j + 3 * kd) % 17) / 17.0f);
            }
        }
        q.cpu_upload();
        k.cpu_upload();
        v.cpu_upload();
    }

    {
        float scale = 1.0f / std::sqrt(float(Dh));
        float16_t* qsp = q_scaled.cpu();
        float16_t* qp = q.cpu();
        for (uint32_t idx = 0; idx < q.shape.count(); ++idx) {
            qsp[idx] = float16_t(float(qp[idx]) * scale);
        }
        q_scaled.cpu_upload();
    }

    float flash_time = 1e9f;
    float attention_time = 1e9f;
    for(int n = 0; n < 1; ++n) {
        const int ITERS = 16;
        for(int it = 0; it < ITERS; ++it) {
            evk::CmdTimestamp("flash_attention", [&]() {
                evk::ai::flash_attention(q, k, v, o_flash);
            });
        }

        for(int it = 0; it < ITERS; ++it) {
            evk::CmdTimestamp("attention", [&]() {
                evk::ai::matmul(q_scaled, k, s, false, true, false, 64, 64);   // S = (Q/√Dh) * K^T
                evk::ai::softmax(s, p);                                        // P = softmax(S) over last dim
                evk::ai::matmul(p, v, o_base, false, false, false, 64, 64);    // O = P * V
            });
        }

        o_flash.cpu_download();
        o_base.cpu_download();

        for(auto& ts : evk::GetTimestamps()) {
            if (strcmp(ts.name, "flash_attention") == 0) {
                flash_time = fmin(flash_time, float(ts.end - ts.start));
            } else if (strcmp(ts.name, "attention") == 0) {
                attention_time = fmin(attention_time, float(ts.end - ts.start));
            }
        }

        bool ok = true;
        float16_t* ofp = o_flash.cpu();
        float16_t* obp = o_base.cpu();
        for (uint32_t i = 0; i < N * D; ++i) {
            float diff = std::abs(float(ofp[i]) - float(obp[i]));
            if (diff > 5e-2f) { ok = false; printf("diff[%u]: %f\n", i, diff); break; }
        }
        TEST(ok);
    }
    printf("  flash_attention: %.3fms\n  attention: %.3fms\n", flash_time, attention_time);
    printf("  attention/B: %.3fms\n", attention_time / float(B));
}

void test_graph_backward() {
    printf("test_graph_backward() \n");
    Graph graph;

    const uint32_t SIZE = 80u;
    Tensor& target = graph.tensor({SIZE, SIZE}).identity(3.0f);

    Tensor& x = graph.tensor({SIZE, SIZE}).identity(1.0f);
    Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& b = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& mul = graph.matmul(w, x);
    Tensor& y = graph.add(mul, b);
    Tensor& loss = graph.mse_loss(y, target);

    float last_loss = 1e9f;
    bool loss_test = true;
    for(int i = 0; i < 100; ++i) {
        graph.eval(true);
        graph.step(0.05f);
        float loss_val = float(loss.item());
        if(loss_val > last_loss) {
            loss_test = false;
        }
        last_loss = loss_val;
    }
    TEST(loss_test);
}

void test_attn_graph() {
    printf("test_attn_graph() \n");
    Graph graph;
    
}

// Gaussian random number using Box-Muller transform
float randn() {
    static bool has_spare = false;
    static float spare;
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    float u, v, s;
    do {
        u = 2.0f * float(rand()) / float(RAND_MAX) - 1.0f;
        v = 2.0f * float(rand()) / float(RAND_MAX) - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = true;
    return u * s;
}

void test_softmax() {
    printf("test_softmax()\n");
    // Shape (B, N, Dlast) -> softmax over Dlast
    const uint32_t B = 2;
    const uint32_t N = 3;
    const uint32_t Dlast = 5;
    Tensor in({B, N, Dlast});
    Tensor out({B, N, Dlast});

    // Fill deterministic values per row to validate probabilities sum to 1
    float16_t* ip = in.cpu();
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t d = 0; d < Dlast; ++d) {
                uint32_t idx = b * (N * Dlast) + n * Dlast + d;
                ip[idx] = float16_t(float(d - 2)); // centered range [-2..2]
            }
        }
    }
    in.cpu_upload();

    evk::ai::softmax(in, out);
    out.cpu_download();
    float16_t* op = out.cpu();

    // Check each row sums to ~1
    bool ok = true;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < Dlast; ++d) {
                uint32_t idx = b * (N * Dlast) + n * Dlast + d;
                sum += float(op[idx]);
            }
            if (std::abs(sum - 1.0f) > 5e-3f) ok = false;
        }
    }
    TEST(ok);
}

void test_adam() {
    printf("test_adam()\n");
    
    // Test basic Adam update: param should move toward reducing gradient
    Tensor param({4});
    Tensor grad({4});
    evk::ai::AdamState state;

    // Initialize param to [1, 2, 3, 4]
    float16_t* pp = param.cpu();
    pp[0] = float16_t(1.0f);
    pp[1] = float16_t(2.0f);
    pp[2] = float16_t(3.0f);
    pp[3] = float16_t(4.0f);
    param.cpu_upload();

    // Gradient pointing "up" - param should decrease
    float16_t* gp = grad.cpu();
    gp[0] = float16_t(0.1f);
    gp[1] = float16_t(0.2f);
    gp[2] = float16_t(0.3f);
    gp[3] = float16_t(0.4f);
    grad.cpu_upload();

    // Store initial values
    float initial_vals[4];
    for (int i = 0; i < 4; ++i) initial_vals[i] = float(pp[i]);

    // Run Adam for a few steps
    for (int step = 0; step < 10; ++step) {
        evk::ai::adam(param, grad, state, 0.1f, 0.9f, 0.999f, 1e-4f);
    }
    evk::Sync();

    param.cpu_download();
    pp = param.cpu();

    // Check that params decreased (moved opposite to positive gradient)
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (float(pp[i]) >= initial_vals[i]) {
            ok = false;
            printf("  param[%d]: %.4f >= initial %.4f\n", i, float(pp[i]), initial_vals[i]);
        }
    }
    TEST(ok);

    // Check that timestep was incremented
    TEST(state.t == 10);
}

void test_adam_convergence() {
    printf("test_adam_convergence()\n");
    
    // Test that Adam can minimize a simple loss function
    // Same setup as test_graph_backward but using Adam
    Graph graph;

    const uint32_t SIZE = 80u;
    Tensor& target = graph.tensor({SIZE, SIZE}).identity(3.0f);

    Tensor& x = graph.tensor({SIZE, SIZE}).identity(1.0f);
    Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& b = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& mul = graph.matmul(w, x);
    Tensor& y = graph.add(mul, b);
    Tensor& loss = graph.mse_loss(y, target);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for(int i = 0; i < 100; ++i) {
        graph.eval(true);
        graph.step_adam(0.01f);  // Use Adam with lr=0.01
        float loss_val = float(loss.item());
        
        if (i == 0) initial_loss = loss_val;
        if (i == 99) final_loss = loss_val;
    }

    printf("  initial_loss: %.6f, final_loss: %.6f\n", initial_loss, final_loss);
    
    // Loss should converge to near zero
    TEST(final_loss < 0.001f);
    // Loss should decrease significantly from initial
    TEST(final_loss < initial_loss);
}

void test_adam_vs_sgd() {
    printf("test_adam_vs_sgd()\n");
    
    // Compare Adam vs SGD convergence speed
    const uint32_t SIZE = 80u;
    const int STEPS = 50;

    // Run with SGD
    float sgd_final_loss;
    {
        Graph graph;
        Tensor& target = graph.tensor({SIZE, SIZE}).identity(3.0f);
        Tensor& x = graph.tensor({SIZE, SIZE}).identity(1.0f);
        Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
        Tensor& b = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
        Tensor& mul = graph.matmul(w, x);
        Tensor& y = graph.add(mul, b);
        Tensor& loss = graph.mse_loss(y, target);

        for(int i = 0; i < STEPS; ++i) {
            graph.eval(true);
            graph.step(0.05f);  // SGD with lr=0.05
        }
        sgd_final_loss = float(loss.item());
    }

    // Run with Adam
    float adam_final_loss;
    {
        Graph graph;
        Tensor& target = graph.tensor({SIZE, SIZE}).identity(3.0f);
        Tensor& x = graph.tensor({SIZE, SIZE}).identity(1.0f);
        Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
        Tensor& b = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
        Tensor& mul = graph.matmul(w, x);
        Tensor& y = graph.add(mul, b);
        Tensor& loss = graph.mse_loss(y, target);

        for(int i = 0; i < STEPS; ++i) {
            graph.eval(true);
            graph.step_adam(0.01f);  // Adam with lr=0.01
        }
        adam_final_loss = float(loss.item());
    }

    printf("  SGD final loss:  %.6f (lr=0.05, %d steps)\n", sgd_final_loss, STEPS);
    printf("  Adam final loss: %.6f (lr=0.01, %d steps)\n", adam_final_loss, STEPS);
    
    // Both should converge to low loss
    TEST(sgd_final_loss < 0.1f);
    TEST(adam_final_loss < 0.1f);
}

void test_adam_batched_matmul() {
    printf("test_adam_batched_matmul()\n");
    
    // Learn W such that X @ W ≈ target for batched data
    // 
    // Task: Learn to produce IDENTITY matrices from batched inputs
    // - Input X: (B, M, K) - each batch has identity matrix (1s on diagonal)
    // - Weight W: (K, N) to learn, starts with random values
    // - Target: (B, M, N) - each batch should be scaled identity (2s on diagonal)
    //
    // X @ W for identity X gives W itself. So W should learn to be 2*I.
    // This tests that Adam can learn a specific weight matrix using batched inputs.
    
    const uint32_t B = 4;   // batch size  
    const uint32_t M = 16;  // rows (tile aligned)
    const uint32_t K = 16;  // inner dim (tile aligned) 
    const uint32_t N = 16;  // cols (tile aligned)
    
    Tensor x({B, M, K});     // batched input - identity matrices
    Tensor w({K, N});        // weight to learn (broadcast across batches)
    Tensor y({B, M, N});     // output
    Tensor target({B, M, N}); // target pattern - scaled identity matrices
    Tensor loss_tensor({1});
    Tensor y_grad({B, M, N});
    Tensor w_grad({K, N});
    
    evk::ai::AdamState adam_state;
    
    // Initialize X with identity matrices for each batch
    {
        float16_t* xp = x.cpu();
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t m = 0; m < M; ++m) {
                for (uint32_t k = 0; k < K; ++k) {
                    uint32_t idx = b * M * K + m * K + k;
                    xp[idx] = float16_t((m == k) ? 1.0f : 0.0f);
                }
            }
        }
        x.cpu_upload();
    }
    
    // Target: identity matrices scaled by 2 (so W should learn to be 2*I)
    {
        float16_t* tp = target.cpu();
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t m = 0; m < M; ++m) {
                for (uint32_t n = 0; n < N; ++n) {
                    uint32_t idx = b * M * N + m * N + n;
                    tp[idx] = float16_t((m == n) ? 2.0f : 0.0f);
                }
            }
        }
        target.cpu_upload();
    }
    
    // Initialize W with small random values
    {
        float16_t* wp = w.cpu();
        for (uint32_t i = 0; i < K * N; ++i) {
            wp[i] = float16_t(0.5f * float(rand()) / float(RAND_MAX));
        }
        w.cpu_upload();
    }
    
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    const int STEPS = 150;
    const float lr = -0.02f;  // Negative because mse_loss gradient convention
    
    for (int step = 0; step < STEPS; ++step) {
        // Forward: y = x @ w (batched matmul with broadcast)
        evk::ai::matmul(x, w, y, false, false, false, 16, 16);
        
        // Compute MSE loss and gradient
        evk::ai::mse_loss(y, target, y_grad, loss_tensor);
        
        // Backward: dW = X^T @ dY (accumulated across batches)
        // First batch: initialize w_grad
        {
            Tensor x_b({M, K});
            Tensor y_grad_b({M, N});
            
            x.cpu_download();
            y_grad.cpu_download();
            float16_t* xp = x.cpu();
            float16_t* ygp = y_grad.cpu();
            float16_t* xbp = x_b.cpu();
            float16_t* ygbp = y_grad_b.cpu();
            
            for (uint32_t i = 0; i < M * K; ++i) xbp[i] = xp[i];
            for (uint32_t i = 0; i < M * N; ++i) ygbp[i] = ygp[i];
            x_b.cpu_upload();
            y_grad_b.cpu_upload();
            
            evk::ai::matmul(x_b, y_grad_b, w_grad, true, false, false, 16, 16);
        }
        
        // Remaining batches: accumulate into w_grad
        for (uint32_t b = 1; b < B; ++b) {
            Tensor x_b({M, K});
            Tensor y_grad_b({M, N});
            
            float16_t* xp = x.cpu();
            float16_t* ygp = y_grad.cpu();
            float16_t* xbp = x_b.cpu();
            float16_t* ygbp = y_grad_b.cpu();
            
            for (uint32_t i = 0; i < M * K; ++i) xbp[i] = xp[b * M * K + i];
            for (uint32_t i = 0; i < M * N; ++i) ygbp[i] = ygp[b * M * N + i];
            x_b.cpu_upload();
            y_grad_b.cpu_upload();
            
            evk::ai::matmul(x_b, y_grad_b, w_grad, true, false, true, 16, 16);
        }
        
        // Adam update
        evk::ai::adam(w, w_grad, adam_state, lr);
        evk::Sync();
        
        loss_tensor.cpu_download();
        float loss_val = float(loss_tensor.cpu()[0]);
        
        if (step == 0) initial_loss = loss_val;
        if (step == STEPS - 1) final_loss = loss_val;
        
        if (step % 30 == 0 || step == STEPS - 1) {
            printf("  step %3d: loss = %.6f\n", step, loss_val);
        }
    }
    
    printf("\n  Initial loss: %.6f\n", initial_loss);
    printf("  Final loss:   %.6f\n", final_loss);
    
    // Visualize final output vs target (batch 0, first 6x6)
    printf("\n  Output (batch 0)         Target (2*Identity)\n");
    y.cpu_download();
    float16_t* yp = y.cpu();
    float16_t* tp = target.cpu();
    for (uint32_t m = 0; m < 6; ++m) {
        printf("  ");
        for (uint32_t n = 0; n < 6; ++n) {
            float v = float(yp[m * N + n]);
            printf("%4.1f ", v);
        }
        printf("   |   ");
        for (uint32_t n = 0; n < 6; ++n) {
            float v = float(tp[m * N + n]);
            printf("%4.1f ", v);
        }
        printf("\n");
    }
    
    // Visualize learned W (should be close to 2*Identity)
    printf("\n  Learned W (first 6x6, should be ~2*I):\n");
    w.cpu_download();
    float16_t* wp = w.cpu();
    for (uint32_t k = 0; k < 6; ++k) {
        printf("  ");
        for (uint32_t n = 0; n < 6; ++n) {
            float v = float(wp[k * N + n]);
            printf("%4.1f ", v);
        }
        printf("\n");
    }
    
    // Test: loss should decrease significantly
    TEST(final_loss < initial_loss * 0.01f);
    // Test: final loss should be very low
    TEST(final_loss < 0.001f);
}
