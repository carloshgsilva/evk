#include <iostream>
#include <evk_ai.h>

#include "win_dbg.h"
#include "bmp.h"

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

void benchmark_matmul() {
    printf("benchmark_matmul()");
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

void test_circle_fitting() {
    printf("test_circle_fitting()\n");
    
    srand(42);  // Reproducible
    
    // ========================================
    // 1. Generate ground truth: TWO circles with noisy points
    // ========================================
    const uint32_t POINTS_PER_CIRCLE = 12;
    const uint32_t NUM_POINTS = POINTS_PER_CIRCLE * 2;
    
    // Circle 1: top-left
    const float TRUE_CX1 = -2.0f;
    const float TRUE_CY1 = 2.0f;
    const float TRUE_R1 = 2.5f;
    
    // Circle 2: bottom-right  
    const float TRUE_CX2 = 3.5f;
    const float TRUE_CY2 = -1.5f;
    const float TRUE_R2 = 2.0f;
    
    const float NOISE = 0.12f;
    
    float points_x[NUM_POINTS];
    float points_y[NUM_POINTS];
    
    // Generate points for circle 1
    for (uint32_t i = 0; i < POINTS_PER_CIRCLE; ++i) {
        float angle = 2.0f * 3.14159265f * float(i) / float(POINTS_PER_CIRCLE);
        float r_noise = TRUE_R1 + randn() * NOISE;
        points_x[i] = TRUE_CX1 + r_noise * cosf(angle) + randn() * NOISE * 0.5f;
        points_y[i] = TRUE_CY1 + r_noise * sinf(angle) + randn() * NOISE * 0.5f;
    }
    
    // Generate points for circle 2
    for (uint32_t i = 0; i < POINTS_PER_CIRCLE; ++i) {
        float angle = 2.0f * 3.14159265f * float(i) / float(POINTS_PER_CIRCLE);
        float r_noise = TRUE_R2 + randn() * NOISE;
        points_x[POINTS_PER_CIRCLE + i] = TRUE_CX2 + r_noise * cosf(angle) + randn() * NOISE * 0.5f;
        points_y[POINTS_PER_CIRCLE + i] = TRUE_CY2 + r_noise * sinf(angle) + randn() * NOISE * 0.5f;
    }
    
    printf("  Ground truth:\n");
    printf("    Circle 1: center=(%.2f, %.2f), radius=%.2f\n", TRUE_CX1, TRUE_CY1, TRUE_R1);
    printf("    Circle 2: center=(%.2f, %.2f), radius=%.2f\n", TRUE_CX2, TRUE_CY2, TRUE_R2);
    
    // ========================================
    // 2. MLP using Graph: input(6) -> hidden(64) -> hidden(64) -> output(1)
    //    Input: [cx1, cy1, r1, cx2, cy2, r2]
    //    Output: predicted loss
    // ========================================
    const uint32_t BATCH_SIZE = 64;
    const uint32_t INPUT_DIM_PAD = 16;   // 6 params padded to 16
    const uint32_t HIDDEN_DIM = 64;
    const uint32_t OUTPUT_DIM_PAD = 16;
    
    auto init_weights = [](Tensor& t, float scale) {
        float16_t* p = t.cpu();
        for (uint32_t i = 0; i < t.shape.count(); ++i) {
            p[i] = float16_t(randn() * scale);
        }
        t.cpu_upload();
    };
    
    // Build MLP graph
    Graph mlp;
    Tensor& input_t = mlp.tensor({BATCH_SIZE, INPUT_DIM_PAD});
    Tensor& w1 = mlp.tensor({INPUT_DIM_PAD, HIDDEN_DIM}, true);
    Tensor& w2 = mlp.tensor({HIDDEN_DIM, HIDDEN_DIM}, true);
    Tensor& w3 = mlp.tensor({HIDDEN_DIM, OUTPUT_DIM_PAD}, true);
    Tensor& target_t = mlp.tensor({BATCH_SIZE, OUTPUT_DIM_PAD});
    
    // Forward: input -> matmul(w1) -> relu -> matmul(w2) -> relu -> matmul(w3) -> mse_loss
    Tensor& h1_pre = mlp.matmul(input_t, w1);
    Tensor& h1 = mlp.relu(h1_pre);
    Tensor& h2_pre = mlp.matmul(h1, w2);
    Tensor& h2 = mlp.relu(h2_pre);
    Tensor& output_t = mlp.matmul(h2, w3);
    Tensor& loss_t = mlp.mse_loss(output_t, target_t);
    
    init_weights(w1, 0.25f);
    init_weights(w2, 0.15f);
    init_weights(w3, 0.15f);
    
    // ========================================
    // 3. Search state: 6 parameters for 2 circles
    // ========================================
    float s_cx1 = 0.0f, s_cy1 = 0.0f, s_r1 = 1.5f;
    float s_cx2 = 1.0f, s_cy2 = 0.0f, s_r2 = 1.5f;
    float search_sigma = 5.0f;
    
    // Best found
    float best_cx1 = s_cx1, best_cy1 = s_cy1, best_r1 = s_r1;
    float best_cx2 = s_cx2, best_cy2 = s_cy2, best_r2 = s_r2;
    
    // Loss: each point assigned to nearest circle
    auto compute_true_loss = [&](float cx1, float cy1, float r1, float cx2, float cy2, float r2) -> float {
        float loss = 0.0f;
        for (uint32_t i = 0; i < NUM_POINTS; ++i) {
            float dx1 = points_x[i] - cx1;
            float dy1 = points_y[i] - cy1;
            float dist1 = fabsf(sqrtf(dx1*dx1 + dy1*dy1) - r1);
            
            float dx2 = points_x[i] - cx2;
            float dy2 = points_y[i] - cy2;
            float dist2 = fabsf(sqrtf(dx2*dx2 + dy2*dy2) - r2);
            
            float min_dist = fminf(dist1, dist2);
            loss += min_dist * min_dist;
        }
        return loss / float(NUM_POINTS);
    };
    
    float best_loss = compute_true_loss(best_cx1, best_cy1, best_r1, best_cx2, best_cy2, best_r2);
    
    printf("  Initial guess:\n");
    printf("    Circle 1: center=(%.2f, %.2f), radius=%.2f\n", s_cx1, s_cy1, s_r1);
    printf("    Circle 2: center=(%.2f, %.2f), radius=%.2f\n", s_cx2, s_cy2, s_r2);
    printf("  Initial loss: %.6f\n", best_loss);
    
    // ========================================
    // Image export helper
    // ========================================
    const int IMG_SIZE = 600;
    float view_min_x = -6.0f, view_max_x = 7.0f;
    float view_min_y = -5.0f, view_max_y = 6.0f;
    
    auto export_image = [&](int iter, float cx1, float cy1, float r1, float cx2, float cy2, float r2, float sigma) {
        BMP img(IMG_SIZE, IMG_SIZE);
        ImageMapper mapper(view_min_x, view_max_x, view_min_y, view_max_y, IMG_SIZE, IMG_SIZE, 10);
        
        img.clear(25, 25, 35);
        img.draw_grid(IMG_SIZE / 12, 40, 40, 50);
        
        // Ground truth circles - green
        img.draw_circle(mapper.to_screen_x(TRUE_CX1), mapper.to_screen_y(TRUE_CY1), 
                        mapper.to_screen_radius(TRUE_R1), 80, 180, 80, 2);
        img.draw_circle(mapper.to_screen_x(TRUE_CX2), mapper.to_screen_y(TRUE_CY2), 
                        mapper.to_screen_radius(TRUE_R2), 80, 180, 80, 2);
        
        // Current estimate circles - cyan and magenta
        img.draw_circle(mapper.to_screen_x(cx1), mapper.to_screen_y(cy1), 
                        mapper.to_screen_radius(r1), 100, 220, 255, 2);
        img.draw_cross(mapper.to_screen_x(cx1), mapper.to_screen_y(cy1), 8, 100, 220, 255);
        
        img.draw_circle(mapper.to_screen_x(cx2), mapper.to_screen_y(cy2), 
                        mapper.to_screen_radius(r2), 255, 120, 200, 2);
        img.draw_cross(mapper.to_screen_x(cx2), mapper.to_screen_y(cy2), 8, 255, 120, 200);
        
        // Points - yellow
        for (uint32_t i = 0; i < NUM_POINTS; ++i) {
            int px = mapper.to_screen_x(points_x[i]);
            int py = mapper.to_screen_y(points_y[i]);
            img.draw_point(px, py, 5, 255, 220, 80);
        }
        
        char filename[128];
        snprintf(filename, sizeof(filename), "build/circle_fit_%03d.bmp", iter);
        img.save(filename);
    };
    
    export_image(0, s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2, search_sigma);
    
    // ========================================
    // 4. Main optimization loop
    // ========================================
    const int OUTER_ITERS = 40;
    const int TRAIN_ITERS = 120;
    const float LR = 0.003f;
    
    for (int outer = 0; outer < OUTER_ITERS; ++outer) {
        // if (outer > 0 && outer % 12 == 0) {
        //     init_weights(w1, 0.25f);
        //     init_weights(w2, 0.15f);
        //     init_weights(w3, 0.15f);
        //     mlp.reset_adam();
        // }
        
        for (int train = 0; train < TRAIN_ITERS; ++train) {
            float16_t* inp = input_t.cpu();
            float16_t* tgt = target_t.cpu();
            
            for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
                float cx1 = s_cx1 + randn() * search_sigma;
                float cy1 = s_cy1 + randn() * search_sigma;
                float r1 = fmaxf(0.5f, s_r1 + randn() * search_sigma * 0.3f);
                float cx2 = s_cx2 + randn() * search_sigma;
                float cy2 = s_cy2 + randn() * search_sigma;
                float r2 = fmaxf(0.5f, s_r2 + randn() * search_sigma * 0.3f);
                
                // Normalize inputs relative to current search center
                inp[b * INPUT_DIM_PAD + 0] = float16_t((cx1 - s_cx1) / search_sigma);
                inp[b * INPUT_DIM_PAD + 1] = float16_t((cy1 - s_cy1) / search_sigma);
                inp[b * INPUT_DIM_PAD + 2] = float16_t((r1 - s_r1) / search_sigma);
                inp[b * INPUT_DIM_PAD + 3] = float16_t((cx2 - s_cx2) / search_sigma);
                inp[b * INPUT_DIM_PAD + 4] = float16_t((cy2 - s_cy2) / search_sigma);
                inp[b * INPUT_DIM_PAD + 5] = float16_t((r2 - s_r2) / search_sigma);
                for (uint32_t j = 6; j < INPUT_DIM_PAD; ++j) {
                    inp[b * INPUT_DIM_PAD + j] = float16_t(0.0f);
                }
                
                float true_loss = compute_true_loss(cx1, cy1, r1, cx2, cy2, r2);
                tgt[b * OUTPUT_DIM_PAD + 0] = float16_t(logf(true_loss + 1.0f));
                for (uint32_t j = 1; j < OUTPUT_DIM_PAD; ++j) {
                    tgt[b * OUTPUT_DIM_PAD + j] = float16_t(0.0f);
                }
                
                if (true_loss < best_loss) {
                    best_loss = true_loss;
                    best_cx1 = cx1; best_cy1 = cy1; best_r1 = r1;
                    best_cx2 = cx2; best_cy2 = cy2; best_r2 = r2;
                }
            }
            input_t.cpu_upload();
            target_t.cpu_upload();
            
            // Forward + Backward using Graph
            mlp.eval(true);
            mlp.step_adam(LR);
            evk::Sync();
        }
        
        // ========================================
        // 5. Get gradient from backward pass at search center
        // ========================================
        // Evaluate at current search center (input = 0 in normalized coords)
        {
            float16_t* inp = input_t.cpu();
            for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
                for (uint32_t j = 0; j < INPUT_DIM_PAD; ++j) {
                    inp[b * INPUT_DIM_PAD + j] = float16_t(0.0f);
                }
            }
            // Set target to 0 (we want to minimize the output)
            float16_t* tgt = target_t.cpu();
            for (uint32_t i = 0; i < target_t.shape.count(); ++i) {
                tgt[i] = float16_t(0.0f);
            }
            input_t.cpu_upload();
            target_t.cpu_upload();
            
            mlp.eval(true);  // Forward + Backward to get input gradients
            evk::Sync();
        }
        
        // Get gradients from input tensor (averaged over batch)
        input_t.grad().cpu_download();
        float16_t* grad_ptr = input_t.grad().cpu();
        float g[6] = {0.0f};
        for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
            for (int i = 0; i < 6; ++i) {
                g[i] += float(grad_ptr[b * INPUT_DIM_PAD + i]);
            }
        }
        for (int i = 0; i < 6; ++i) g[i] /= float(BATCH_SIZE);
        
        float grad_norm = 0.0f;
        for (int i = 0; i < 6; ++i) grad_norm += g[i] * g[i];
        grad_norm = sqrtf(grad_norm);
        
        if (grad_norm > 1e-6f) {
            float step = search_sigma * 0.25f;
            s_cx1 -= step * g[0] / grad_norm;
            s_cy1 -= step * g[1] / grad_norm;
            s_r1  -= step * g[2] / grad_norm;
            s_cx2 -= step * g[3] / grad_norm;
            s_cy2 -= step * g[4] / grad_norm;
            s_r2  -= step * g[5] / grad_norm;
        }
        s_r1 = fmaxf(0.5f, s_r1);
        s_r2 = fmaxf(0.5f, s_r2);
        
        // Blend toward best
        s_cx1 = 0.7f * s_cx1 + 0.3f * best_cx1;
        s_cy1 = 0.7f * s_cy1 + 0.3f * best_cy1;
        s_r1  = 0.7f * s_r1  + 0.3f * best_r1;
        s_cx2 = 0.7f * s_cx2 + 0.3f * best_cx2;
        s_cy2 = 0.7f * s_cy2 + 0.3f * best_cy2;
        s_r2  = 0.7f * s_r2  + 0.3f * best_r2;
        
        search_sigma *= 0.93f;
        
        float true_loss = compute_true_loss(s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2);
        if (outer % 5 == 0 || outer == OUTER_ITERS - 1) {
            printf("  [iter %2d] C1=(%.2f,%.2f,%.2f) C2=(%.2f,%.2f,%.2f) loss=%.4f sigma=%.2f\n",
                   outer, s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2, true_loss, search_sigma);
            export_image(outer + 1, s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2, search_sigma);
        }
    }
    
    // Use best found
    s_cx1 = best_cx1; s_cy1 = best_cy1; s_r1 = best_r1;
    s_cx2 = best_cx2; s_cy2 = best_cy2; s_r2 = best_r2;
    
    export_image(OUTER_ITERS + 1, s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2, 0.0f);
    printf("  Exported images to build/circle_fit_*.bmp\n");
    
    // ========================================
    // 6. Final results - match circles to ground truth
    // ========================================
    float final_loss = compute_true_loss(s_cx1, s_cy1, s_r1, s_cx2, s_cy2, s_r2);
    
    // Determine which estimate matches which ground truth
    float dist_1to1 = sqrtf((s_cx1-TRUE_CX1)*(s_cx1-TRUE_CX1) + (s_cy1-TRUE_CY1)*(s_cy1-TRUE_CY1));
    float dist_1to2 = sqrtf((s_cx1-TRUE_CX2)*(s_cx1-TRUE_CX2) + (s_cy1-TRUE_CY2)*(s_cy1-TRUE_CY2));
    
    float e_cx1, e_cy1, e_r1, e_cx2, e_cy2, e_r2;
    if (dist_1to1 < dist_1to2) {
        e_cx1 = fabsf(s_cx1 - TRUE_CX1); e_cy1 = fabsf(s_cy1 - TRUE_CY1); e_r1 = fabsf(s_r1 - TRUE_R1);
        e_cx2 = fabsf(s_cx2 - TRUE_CX2); e_cy2 = fabsf(s_cy2 - TRUE_CY2); e_r2 = fabsf(s_r2 - TRUE_R2);
    } else {
        e_cx1 = fabsf(s_cx2 - TRUE_CX1); e_cy1 = fabsf(s_cy2 - TRUE_CY1); e_r1 = fabsf(s_r2 - TRUE_R1);
        e_cx2 = fabsf(s_cx1 - TRUE_CX2); e_cy2 = fabsf(s_cy1 - TRUE_CY2); e_r2 = fabsf(s_r1 - TRUE_R2);
    }
    
    printf("\n  Final estimates:\n");
    printf("    Circle 1: center=(%.3f, %.3f), radius=%.3f\n", s_cx1, s_cy1, s_r1);
    printf("    Circle 2: center=(%.3f, %.3f), radius=%.3f\n", s_cx2, s_cy2, s_r2);
    printf("  Final loss: %.6f\n", final_loss);
    printf("  Errors: C1(%.3f,%.3f,%.3f) C2(%.3f,%.3f,%.3f)\n", e_cx1, e_cy1, e_r1, e_cx2, e_cy2, e_r2);
    
    float max_error = fmaxf(fmaxf(fmaxf(e_cx1, e_cy1), fmaxf(e_r1, e_cx2)), fmaxf(e_cy2, e_r2));
    TEST(max_error < 1.0f);
    TEST(final_loss < 0.2f);
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

void test_next_token_prediction() {
    printf("test_next_token_prediction()\n");
    
    // ========================================
    // 1. Vocabulary and tokenization
    // ========================================
    const char* VOCAB = " abcdefghijklmnopqrstuvwxyz.,\n";
    const uint32_t VOCAB_SIZE = 32;
    const uint32_t ACTUAL_VOCAB = 30;
    
    auto char_to_token = [&](char c) -> uint16_t {
        for (uint32_t i = 0; VOCAB[i]; ++i) {
            if (VOCAB[i] == c) return uint16_t(i);
        }
        return 0;
    };
    
    auto token_to_char = [&](uint16_t t) -> char {
        if (t < ACTUAL_VOCAB) return VOCAB[t];
        return '?';
    };
    
    // ========================================
    // 2. Lorem ipsum training samples
    // ========================================
    const char* samples[] = {
        "lorem ipsum dolor sit amet.",
        "consectetur adipiscing elit.",
        "sed do eiusmod tempor incid.",
        "incididunt ut labore dolore.",
        "dolore magna aliqua ut enim.",
        "enim ad minim veniam quis.",
        "quis nostrud exercitation.",
        "ullamco laboris nisi aliquip.",
    };
    const uint32_t NUM_SAMPLES = sizeof(samples) / sizeof(samples[0]);
    
    // ========================================
    // 3. Model hyperparameters (simplified 2-layer MLP with causal context)
    // ========================================
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    
    printf("  Model: vocab=%u, seq=%u, embed=%u, hidden=%u (using Graph API)\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    
    // ========================================
    // 4. Build computation graph using Graph API
    // ========================================
    Graph model;
    
    // Input tokens (filled each batch) - indices as uint16
    Tensor& input_tokens = model.tensor({BATCH_SIZE * SEQ_LEN});
    Tensor& targets = model.tensor({BATCH_SIZE * SEQ_LEN});
    
    // Embedding parameters (learnable)
    Tensor& token_emb = model.tensor({VOCAB_SIZE, EMBED_DIM}, true);
    Tensor& pos_emb = model.tensor({SEQ_LEN, EMBED_DIM}, true);
    
    // Model parameters (learnable)
    Tensor& w1 = model.tensor({EMBED_DIM, HIDDEN_DIM}, true);
    Tensor& w2 = model.tensor({HIDDEN_DIM, EMBED_DIM}, true);
    Tensor& w_out = model.tensor({EMBED_DIM, VOCAB_SIZE}, true);
    
    // Build forward graph:
    // tokens -> embed(token_emb) -> add_position_embedding(pos_emb) -> Linear(w1) -> ReLU -> Linear(w2) -> residual -> Linear(w_out) -> logits
    
    // Token embedding lookup
    Tensor& embedded = model.embed(token_emb, input_tokens);
    
    // Add positional embeddings (broadcast across batch)
    Tensor& input_with_pos = model.add_position_embedding(embedded, pos_emb, BATCH_SIZE, SEQ_LEN);
    
    // MLP layers
    Tensor& hidden = model.matmul(input_with_pos, w1);
    Tensor& hidden_relu = model.relu(hidden);
    Tensor& hidden_proj = model.matmul(hidden_relu, w2);
    Tensor& residual = model.add(input_with_pos, hidden_proj);
    Tensor& logits = model.matmul(residual, w_out);
    Tensor& loss = model.cross_entropy_loss(logits, targets);
    
    // ========================================
    // 5. Initialize weights using Tensor::random_init()
    // ========================================
    float scale = 0.1f / sqrtf(float(EMBED_DIM));
    token_emb.random_init(0.1f);
    pos_emb.random_init(0.1f);
    w1.random_init(scale);
    w2.random_init(scale);
    w_out.random_init(scale);
    
    // ========================================
    // 6. Training loop using Graph API
    // ========================================
    const int EPOCHS = 200;
    const float LR = 0.01f;
    
    printf("  Training for %d epochs...\n", EPOCHS);
    
    float initial_loss = 0.0f;
    float last_loss = 1e9f;
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (uint32_t batch_start = 0; batch_start + BATCH_SIZE <= NUM_SAMPLES; batch_start += BATCH_SIZE) {
            // Prepare batch data - fill input_tokens and targets
            uint16_t* inp = (uint16_t*)input_tokens.cpu();
            uint16_t* tgt = (uint16_t*)targets.cpu();
            
            for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
                const char* sample = samples[batch_start + b];
                uint32_t len = (uint32_t)strlen(sample);
                
                for (uint32_t t = 0; t < SEQ_LEN; ++t) {
                    inp[b * SEQ_LEN + t] = (t < len) ? char_to_token(sample[t]) : uint16_t(0);
                    tgt[b * SEQ_LEN + t] = (t + 1 < len) ? char_to_token(sample[t + 1]) : uint16_t(0);
                }
            }
            input_tokens.cpu_upload();
            targets.cpu_upload();
            
            // Forward + Backward pass using Graph API
            // Embeddings are now part of the graph!
            model.eval(true);
            evk::Sync();
            
            loss.cpu_download();
            float loss_val = float(loss.cpu()[0]);
            epoch_loss += loss_val;
            num_batches++;
            
            if (epoch == 0 && batch_start == 0) {
                initial_loss = loss_val;
            }
            
            // Optimizer step for ALL graph parameters (including embeddings)
            // cross_entropy gradient is standard (softmax - one_hot), needs positive lr
            model.step_adam(-LR);
            
            evk::Sync();
        }
        
        float avg_loss = epoch_loss / float(num_batches);
        
        if (epoch % 40 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, avg_loss);
        }
        
        last_loss = avg_loss;
    }
    
    // ========================================
    // 7. Autoregressive generation (using Graph forward pass)
    // ========================================
    printf("\n  Autoregressive generation:\n");
    
    const char* prompts[] = {"lorem ", "dolor ", "sed do"};
    const int NUM_PROMPTS = 3;
    const int GEN_LEN = 20;
    
    for (int p = 0; p < NUM_PROMPTS; ++p) {
        char generated[64] = {0};
        strcpy(generated, prompts[p]);
        uint32_t cur_len = (uint32_t)strlen(generated);
        
        uint16_t tokens[SEQ_LEN] = {0};
        for (uint32_t i = 0; i < cur_len && i < SEQ_LEN; ++i) {
            tokens[i] = char_to_token(generated[i]);
        }
        
        for (int g = 0; g < GEN_LEN && cur_len < SEQ_LEN - 1; ++g) {
            // Fill input_tokens with current sequence
            uint16_t* inp = (uint16_t*)input_tokens.cpu();
            memset(inp, 0, BATCH_SIZE * SEQ_LEN * sizeof(uint16_t));
            for (uint32_t i = 0; i < cur_len; ++i) {
                inp[i] = tokens[i];
            }
            input_tokens.cpu_upload();
            
            // Forward pass only (embeddings are part of graph)
            model.eval(false);
            evk::Sync();
            
            // Greedy sampling: pick token with highest logit at position cur_len-1
            logits.cpu_download();
            float16_t* lp = logits.cpu();
            uint32_t pos = cur_len - 1;
            
            float max_val = -1e9f;
            uint16_t next_token = 0;
            for (uint32_t voc = 0; voc < ACTUAL_VOCAB; ++voc) {
                float val = float(lp[pos * VOCAB_SIZE + voc]);
                if (val > max_val) {
                    max_val = val;
                    next_token = uint16_t(voc);
                }
            }
            
            tokens[cur_len] = next_token;
            generated[cur_len] = token_to_char(next_token);
            cur_len++;
        }
        
        generated[cur_len] = '\0';
        printf("    \"%s\" -> \"%s\"\n", prompts[p], generated);
    }
    
    // ========================================
    // 8. Validation
    // ========================================
    printf("\n  Validation:\n");
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initial_loss, last_loss);
    
    TEST(last_loss < initial_loss);
    TEST(last_loss < 2.5f);
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

int main() {
    set_unhandled_exception_filter();

    evk::InitializeEVK({
        .applicationName = "evk_example",
        .applicationVersion = 1,
        .engineName = "evk_example_engine",
        .engineVersion = 1,
        .enableSwapchain = false,
    });

    evk::ai::initialize();

    // test_add();
    // test_matmul();
    // test_matmul_broadcast();
    // test_mse_loss();
    // test_flash_attention_backward();
    // test_flash_attention_cmp_softmax();
    // test_softmax();
    // test_graph_backward();

    // test_adam();
    // test_adam_convergence();
    // test_adam_vs_sgd();
    // test_adam_batched_matmul();

    // test_circle_fitting();

    // benchmark_matmul();

    test_next_token_prediction();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
