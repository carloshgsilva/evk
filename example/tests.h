
#include <evk_ai.h>

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

static bool approx_eq(float a, float b, float tol = 1e-2f) {
    return std::abs(a - b) <= tol;
}

static float gelu_approx_cpu(float x) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta = 0.044715f;
    float x3 = x * x * x;
    float u = kAlpha * (x + kBeta * x3);
    return 0.5f * x * (1.0f + std::tanh(u));
}

static float gelu_grad_approx_cpu(float x) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float u = kAlpha * (x + kBeta * x3);
    float t = std::tanh(u);
    float sech2 = 1.0f - t * t;
    float du_dx = kAlpha * (1.0f + 3.0f * kBeta * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
}

static float quantize_fp16_cpu(float x) {
    return float(float16_t(x));
}

static void upload_tensor_from_f32(Tensor& tensor, const std::vector<float>& values) {
    assert(tensor.shape.count() == values.size());
    float16_t* dst = tensor.cpu();
    for (uint32_t i = 0; i < tensor.shape.count(); ++i) {
        dst[i] = float16_t(values[i]);
    }
    tensor.cpu_upload();
}

static void download_tensor_to_f32(Tensor& tensor, std::vector<float>& values) {
    tensor.cpu_download();
    float16_t* src = tensor.cpu();
    values.resize(tensor.shape.count());
    for (uint32_t i = 0; i < tensor.shape.count(); ++i) {
        values[i] = float(src[i]);
    }
}

static void cpu_matmul_nn(const std::vector<float>& a,
                          const std::vector<float>& b,
                          std::vector<float>& c,
                          uint32_t rows,
                          uint32_t inner,
                          uint32_t cols) {
    c.assign(rows * cols, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t n = 0; n < cols; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < inner; ++k) {
                sum += a[r * inner + k] * b[k * cols + n];
            }
            c[r * cols + n] = sum;
        }
    }
}

static void cpu_matmul_tn(const std::vector<float>& a,
                          const std::vector<float>& b,
                          std::vector<float>& c,
                          uint32_t rows,
                          uint32_t inner,
                          uint32_t cols) {
    c.assign(inner * cols, 0.0f);
    for (uint32_t k = 0; k < inner; ++k) {
        for (uint32_t n = 0; n < cols; ++n) {
            float sum = 0.0f;
            for (uint32_t r = 0; r < rows; ++r) {
                sum += a[r * inner + k] * b[r * cols + n];
            }
            c[k * cols + n] = sum;
        }
    }
}

static float cpu_mse_loss_and_grad(const std::vector<float>& predicted,
                                   const std::vector<float>& target,
                                   std::vector<float>& grad) {
    assert(predicted.size() == target.size());
    grad.resize(predicted.size());
    double loss_sum = 0.0;
    float grad_scale = 2.0f / float(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        float diff = predicted[i] - target[i];
        loss_sum += double(diff) * double(diff);
        grad[i] = grad_scale * diff;
    }
    return float(loss_sum / double(predicted.size()));
}

static void cpu_adam_step(std::vector<float>& param,
                          const std::vector<float>& grad,
                          std::vector<float>& m,
                          std::vector<float>& v,
                          uint32_t timestep,
                          float learning_rate,
                          float beta1,
                          float beta2,
                          float epsilon) {
    assert(param.size() == grad.size());
    assert(m.size() == grad.size());
    assert(v.size() == grad.size());

    float beta1_correction_inv = 1.0f / (1.0f - std::pow(beta1, float(timestep)));
    float beta2_correction_inv = 1.0f / (1.0f - std::pow(beta2, float(timestep)));

    for (size_t i = 0; i < param.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        float m_hat = m[i] * beta1_correction_inv;
        float v_hat = v[i] * beta2_correction_inv;
        param[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

static float rmse_between(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum_sq += diff * diff;
    }
    return float(std::sqrt(sum_sq / double(a.size())));
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
    // dL/dpred = 2*(pred-target)/N
    TEST(std::abs(float(grad_cpu[0]) - 0.0f) < 5e-4f);
    TEST(std::abs(float(grad_cpu[1]) - 0.0f) < 5e-4f);
    float expected_g2 = 2.0f * (3.0f - 4.0f) / 3.0f; // -2/3
    TEST(std::abs(float(grad_cpu[2]) - expected_g2) < 5e-3f);
}

void test_cross_entropy_loss() {
    printf("test_cross_entropy_loss()\n");

    const uint32_t POSITIONS = 3;
    const uint32_t VOCAB = 4;

    Tensor logits({POSITIONS, VOCAB});
    Tensor targets({POSITIONS});
    Tensor grad({POSITIONS, VOCAB});
    Tensor loss({1});

    float host_logits[POSITIONS][VOCAB] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 2.0f, 0.0f},
        {1.0f, -1.0f, 0.5f, 0.0f}, // ignored row
    };
    uint16_t host_targets[POSITIONS] = {1u, 3u, 0u}; // last target is IGNORE

    float16_t* lp = logits.cpu();
    for (uint32_t i = 0; i < POSITIONS; ++i) {
        for (uint32_t j = 0; j < VOCAB; ++j) {
            lp[i * VOCAB + j] = float16_t(host_logits[i][j]);
        }
    }
    float16_t* tp = targets.cpu();
    for (uint32_t i = 0; i < POSITIONS; ++i) {
        tp[i].value = host_targets[i];
    }
    logits.cpu_upload();
    targets.cpu_upload();

    evk::ai::cross_entropy_loss(logits, targets, grad, loss);

    grad.cpu_download();
    loss.cpu_download();

    float expected_grad[POSITIONS][VOCAB] = {};
    float total_loss = 0.0f;
    uint32_t valid = 0;
    for (uint32_t row = 0; row < POSITIONS; ++row) {
        uint16_t t = host_targets[row];
        if (t == 0u || t >= VOCAB) {
            continue;
        }
        ++valid;
        float row_max = host_logits[row][0];
        for (uint32_t j = 1; j < VOCAB; ++j) {
            row_max = (std::max)(row_max, host_logits[row][j]);
        }
        float denom = 0.0f;
        for (uint32_t j = 0; j < VOCAB; ++j) {
            denom += std::exp(host_logits[row][j] - row_max);
        }
        float log_denom = std::log(denom);
        for (uint32_t j = 0; j < VOCAB; ++j) {
            float prob = std::exp(host_logits[row][j] - row_max) / denom;
            float g = prob;
            if (j == t) {
                g -= 1.0f;
                total_loss += -(host_logits[row][j] - row_max - log_denom);
            }
            expected_grad[row][j] = g;
        }
    }

    float inv_valid = (valid > 0) ? (1.0f / float(valid)) : 0.0f;
    for (uint32_t i = 0; i < POSITIONS; ++i) {
        for (uint32_t j = 0; j < VOCAB; ++j) {
            expected_grad[i][j] *= inv_valid;
        }
    }
    float expected_loss = total_loss * inv_valid;

    const float loss_tol = 5e-3f;
    const float grad_tol = 1e-2f;

    float got_loss = float(loss.cpu()[0]);
    bool loss_ok = std::abs(got_loss - expected_loss) < loss_tol;

    float16_t* gp = grad.cpu();
    bool grad_ok = true;
    for (uint32_t i = 0; i < POSITIONS && grad_ok; ++i) {
        for (uint32_t j = 0; j < VOCAB; ++j) {
            float g = float(gp[i * VOCAB + j]);
            if (std::abs(g - expected_grad[i][j]) > grad_tol) {
                grad_ok = false;
                break;
            }
        }
    }

    TEST(loss_ok);
    TEST(grad_ok);

    // All positions ignored -> zero loss and zero gradients
    {
        const uint32_t POS2 = 2;
        const uint32_t VOCAB2 = 3;
        Tensor logits2({POS2, VOCAB2});
        Tensor targets2({POS2});
        Tensor grad2({POS2, VOCAB2});
        Tensor loss2({1});

        float16_t* l2 = logits2.cpu();
        for (uint32_t i = 0; i < POS2 * VOCAB2; ++i) {
            l2[i] = float16_t(0.1f * float(i));
        }
        float16_t* t2 = targets2.cpu();
        for (uint32_t i = 0; i < POS2; ++i) {
            t2[i].value = 0u;
        }
        logits2.cpu_upload();
        targets2.cpu_upload();

        evk::ai::cross_entropy_loss(logits2, targets2, grad2, loss2);

        grad2.cpu_download();
        loss2.cpu_download();

        float loss_zero = float(loss2.cpu()[0]);
        bool loss_zero_ok = std::abs(loss_zero) < 1e-6f;

        float16_t* g2 = grad2.cpu();
        bool grad_zero_ok = true;
        for (uint32_t i = 0; i < POS2 * VOCAB2; ++i) {
            if (std::abs(float(g2[i])) > 1e-6f) {
                grad_zero_ok = false;
                break;
            }
        }

        TEST(loss_zero_ok);
        TEST(grad_zero_ok);
    }
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
        auto& cmd = evk::ai::GetCmd();
        for(int it = 0; it < ITERS; ++it) {
            cmd.timestamp("flash_attention", [&]() {
                evk::ai::flash_attention(q, k, v, o_flash);
            });
        }

        for(int it = 0; it < ITERS; ++it) {
            cmd.timestamp("attention", [&]() {
                evk::ai::matmul(q_scaled, k, s, false, true, false, 64, 64);   // S = (Q/√Dh) * K^T
                evk::ai::softmax(s, p);                                        // P = softmax(S) over last dim
                evk::ai::matmul(p, v, o_base, false, false, false, 64, 64);    // O = P * V
            });
        }
        evk::ai::SubmitCmd(true);

        o_flash.cpu_download();
        o_base.cpu_download();

        for(const auto& ts : evk::CmdTimestamps()) {
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

    Tensor masked_in({1, 4});
    Tensor masked_out({1, 4});
    float16_t* mp = masked_in.cpu();
    mp[0] = float16_t(0.0f);
    mp[1] = float16_t(-65504.0f);
    mp[2] = float16_t(1.0f);
    mp[3] = float16_t(-65504.0f);
    masked_in.cpu_upload();

    evk::ai::softmax(masked_in, masked_out, 1e-5f);
    masked_out.cpu_download();

    float p0 = float(masked_out.cpu()[0]);
    float p1 = float(masked_out.cpu()[1]);
    float p2 = float(masked_out.cpu()[2]);
    float p3 = float(masked_out.cpu()[3]);
    bool masked_ok = approx_eq(p1, 0.0f, 1e-7f) &&
                     approx_eq(p3, 0.0f, 1e-7f) &&
                     approx_eq(p0 + p2, 1.0f, 5e-3f) &&
                     p0 > 0.49f && p2 > 0.49f;
    TEST(masked_ok);
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
    evk::ai::SubmitCmd(true);

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

void test_adam_fp16_matmul_convergence_vs_cpu_fp32() {
    printf("test_adam_fp16_matmul_convergence_vs_cpu_fp32()\n");

    constexpr uint32_t SIZE = 16u;
    constexpr int STEPS = 96;
    constexpr float kLearningRate = 0.03f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.999f;
    constexpr float kEpsilon = 1e-4f;

    std::vector<float> x_host(SIZE * SIZE);
    std::vector<float> w_init(SIZE * SIZE);
    std::vector<float> w_target(SIZE * SIZE);
    for (uint32_t r = 0; r < SIZE; ++r) {
        for (uint32_t c = 0; c < SIZE; ++c) {
            uint32_t idx = r * SIZE + c;

            float x_val = (r == c ? 1.0f : 0.0f) +
                0.03125f * float(int((r * 5u + c * 3u) % 5u) - 2);
            float w_init_val =
                0.0625f * float(int((r * 7u + c * 11u) % 7u) - 3);
            float w_target_val =
                (r == c ? 0.75f : 0.0f) +
                0.0625f * float(int((r * 13u + c * 9u) % 9u) - 4);

            x_host[idx] = quantize_fp16_cpu(x_val);
            w_init[idx] = quantize_fp16_cpu(w_init_val);
            w_target[idx] = quantize_fp16_cpu(w_target_val);
        }
    }

    std::vector<float> target_host;
    cpu_matmul_nn(x_host, w_target, target_host, SIZE, SIZE, SIZE);
    for (float& value : target_host) {
        value = quantize_fp16_cpu(value);
    }

    Tensor x({SIZE, SIZE});
    Tensor w_gpu({SIZE, SIZE});
    Tensor y_gpu({SIZE, SIZE});
    Tensor target({SIZE, SIZE});
    Tensor loss_gpu_tensor({1});
    Tensor y_grad_gpu({SIZE, SIZE});
    Tensor w_grad_gpu({SIZE, SIZE});
    evk::ai::AdamState gpu_state;

    upload_tensor_from_f32(x, x_host);
    upload_tensor_from_f32(w_gpu, w_init);
    upload_tensor_from_f32(target, target_host);

    std::vector<float> w_cpu = w_init;
    std::vector<float> m_cpu(SIZE * SIZE, 0.0f);
    std::vector<float> v_cpu(SIZE * SIZE, 0.0f);
    std::vector<float> y_cpu;
    std::vector<float> y_grad_cpu;
    std::vector<float> w_grad_cpu;

    float gpu_initial_loss = 0.0f;
    float gpu_final_loss = 0.0f;
    float cpu_initial_loss = 0.0f;
    float cpu_final_loss = 0.0f;
    bool gpu_losses_finite = true;
    bool cpu_losses_finite = true;

    for (int step = 1; step <= STEPS; ++step) {
        evk::ai::matmul(x, w_gpu, y_gpu, false, false, false, 16, 16);
        evk::ai::mse_loss(y_gpu, target, y_grad_gpu, loss_gpu_tensor);
        evk::ai::matmul(x, y_grad_gpu, w_grad_gpu, true, false, false, 16, 16);
        evk::ai::adam(w_gpu, w_grad_gpu, gpu_state, kLearningRate, kBeta1, kBeta2, kEpsilon);
        loss_gpu_tensor.cpu_download(false);
        evk::ai::SubmitCmd(true);

        float gpu_loss = float(loss_gpu_tensor.cpu()[0]);
        gpu_losses_finite = gpu_losses_finite && std::isfinite(gpu_loss);

        cpu_matmul_nn(x_host, w_cpu, y_cpu, SIZE, SIZE, SIZE);
        float cpu_loss = cpu_mse_loss_and_grad(y_cpu, target_host, y_grad_cpu);
        cpu_matmul_tn(x_host, y_grad_cpu, w_grad_cpu, SIZE, SIZE, SIZE);
        cpu_adam_step(w_cpu, w_grad_cpu, m_cpu, v_cpu,
                      uint32_t(step), kLearningRate, kBeta1, kBeta2, kEpsilon);
        cpu_losses_finite = cpu_losses_finite && std::isfinite(cpu_loss);

        if (step == 1) {
            gpu_initial_loss = gpu_loss;
            cpu_initial_loss = cpu_loss;
        }
        if (step == STEPS) {
            gpu_final_loss = gpu_loss;
            cpu_final_loss = cpu_loss;
        }
    }

    std::vector<float> w_gpu_final;
    download_tensor_to_f32(w_gpu, w_gpu_final);

    float gpu_to_target_rmse = rmse_between(w_gpu_final, w_target);
    float cpu_to_target_rmse = rmse_between(w_cpu, w_target);
    float gpu_to_cpu_rmse = rmse_between(w_gpu_final, w_cpu);

    printf("  gpu_loss: %.6f -> %.6f\n", gpu_initial_loss, gpu_final_loss);
    printf("  cpu_loss: %.6f -> %.6f\n", cpu_initial_loss, cpu_final_loss);
    printf("  rmse(target): gpu=%.6f cpu=%.6f gpu_vs_cpu=%.6f\n",
           gpu_to_target_rmse, cpu_to_target_rmse, gpu_to_cpu_rmse);

    TEST(gpu_losses_finite);
    TEST(cpu_losses_finite);
    TEST(gpu_final_loss < gpu_initial_loss * 0.05f);
    TEST(gpu_final_loss < 1e-4f);
    TEST(cpu_final_loss < cpu_initial_loss * 0.02f);
    TEST(gpu_final_loss <= cpu_final_loss + 1e-4f);
    TEST(gpu_to_target_rmse <= cpu_to_target_rmse + 2e-3f);
    TEST(gpu_to_cpu_rmse < 5e-3f);
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
    evk::ai::SubmitCmd(true);
        
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

void test_sgd() {
    printf("test_sgd()\n");

    Tensor param({4});
    Tensor grad({4});

    float16_t* pp = param.cpu();
    float16_t* gp = grad.cpu();
    pp[0] = float16_t(1.0f);
    pp[1] = float16_t(-2.0f);
    pp[2] = float16_t(0.5f);
    pp[3] = float16_t(4.0f);
    gp[0] = float16_t(0.5f);
    gp[1] = float16_t(-1.0f);
    gp[2] = float16_t(2.0f);
    gp[3] = float16_t(0.0f);
    param.cpu_upload();
    grad.cpu_upload();

    evk::ai::sgd(param, grad, 0.1f);
    param.cpu_download();

    TEST(approx_eq(float(param.cpu()[0]), 0.95f, 2e-2f));
    TEST(approx_eq(float(param.cpu()[1]), -1.9f, 2e-2f));
    TEST(approx_eq(float(param.cpu()[2]), 0.3f, 2e-2f));
    TEST(approx_eq(float(param.cpu()[3]), 4.0f, 2e-2f));
}

void test_relu() {
    printf("test_relu()\n");

    Tensor input({4});
    Tensor output({4});
    Tensor grad_out({4});
    Tensor grad_in({4});

    float16_t* ip = input.cpu();
    ip[0] = float16_t(-2.0f);
    ip[1] = float16_t(0.0f);
    ip[2] = float16_t(1.5f);
    ip[3] = float16_t(3.0f);
    input.cpu_upload();
    grad_out.fill(float16_t(1.0f));
    grad_in.fill(float16_t(0.0f));

    evk::ai::relu(input, output);
    output.cpu_download();
    TEST(approx_eq(float(output.cpu()[0]), 0.0f));
    TEST(approx_eq(float(output.cpu()[1]), 0.0f));
    TEST(approx_eq(float(output.cpu()[2]), 1.5f));
    TEST(approx_eq(float(output.cpu()[3]), 3.0f));

    evk::ai::relu_backward(grad_out, input, grad_in);
    grad_in.cpu_download();
    TEST(approx_eq(float(grad_in.cpu()[0]), 0.0f));
    TEST(approx_eq(float(grad_in.cpu()[1]), 0.0f));
    TEST(approx_eq(float(grad_in.cpu()[2]), 1.0f));
    TEST(approx_eq(float(grad_in.cpu()[3]), 1.0f));
}

void test_gelu() {
    printf("test_gelu()\n");

    Tensor input({3});
    Tensor output({3});
    Tensor grad_out({3});
    Tensor grad_in({3});

    float values[3] = {-1.0f, 0.0f, 1.0f};
    float16_t* ip = input.cpu();
    for (uint32_t i = 0; i < 3; ++i) {
        ip[i] = float16_t(values[i]);
    }
    input.cpu_upload();
    grad_out.fill(float16_t(1.0f));
    grad_in.fill(float16_t(0.0f));

    evk::ai::gelu(input, output);
    output.cpu_download();
    for (uint32_t i = 0; i < 3; ++i) {
        TEST(approx_eq(float(output.cpu()[i]), gelu_approx_cpu(values[i]), 2e-2f));
    }

    evk::ai::gelu_backward(grad_out, input, grad_in);
    grad_in.cpu_download();
    for (uint32_t i = 0; i < 3; ++i) {
        TEST(approx_eq(float(grad_in.cpu()[i]), gelu_grad_approx_cpu(values[i]), 3e-2f));
    }
}

void test_scale_zero() {
    printf("test_scale_zero()\n");

    Tensor tensor({4});
    float16_t* tp = tensor.cpu();
    tp[0] = float16_t(2.0f);
    tp[1] = float16_t(-4.0f);
    tp[2] = float16_t(1.0f);
    tp[3] = float16_t(0.5f);
    tensor.cpu_upload();

    evk::ai::scale(tensor, 0.5f);
    tensor.cpu_download();
    TEST(approx_eq(float(tensor.cpu()[0]), 1.0f));
    TEST(approx_eq(float(tensor.cpu()[1]), -2.0f));
    TEST(approx_eq(float(tensor.cpu()[2]), 0.5f));
    TEST(approx_eq(float(tensor.cpu()[3]), 0.25f, 2e-2f));

    evk::ai::zero(tensor);
    tensor.cpu_download();
    for (uint32_t i = 0; i < 4; ++i) {
        TEST(approx_eq(float(tensor.cpu()[i]), 0.0f));
    }
}

void test_embed_and_backward() {
    printf("test_embed_and_backward()\n");

    const uint32_t vocab_size = 4u;
    const uint32_t embed_dim = 3u;

    Tensor embeddings({vocab_size, embed_dim});
    Tensor indices({2});
    Tensor output({2, embed_dim});
    Tensor grad_out({2, embed_dim});
    Tensor grad_embeddings({vocab_size, embed_dim});

    float16_t* ep = embeddings.cpu();
    for (uint32_t i = 0; i < vocab_size * embed_dim; ++i) {
        ep[i] = float16_t(float(i + 1));
    }
    embeddings.cpu_upload();

    float16_t* ip = indices.cpu();
    ip[0].value = 1u;
    ip[1].value = 3u;
    indices.cpu_upload();

    evk::ai::embed(embeddings, indices, output);
    output.cpu_download();
    for (uint32_t d = 0; d < embed_dim; ++d) {
        TEST(approx_eq(float(output.cpu()[d]), float(ep[embed_dim + d])));
        TEST(approx_eq(float(output.cpu()[embed_dim + d]), float(ep[3u * embed_dim + d])));
    }

    float16_t* gop = grad_out.cpu();
    for (uint32_t i = 0; i < 2u * embed_dim; ++i) {
        gop[i] = float16_t(float(i + 1));
    }
    grad_out.cpu_upload();
    grad_embeddings.fill(float16_t(0.0f));

    evk::ai::embed_backward(grad_out, indices, grad_embeddings);
    grad_embeddings.cpu_download();

    for (uint32_t token = 0; token < vocab_size; ++token) {
        for (uint32_t d = 0; d < embed_dim; ++d) {
            float expected = 0.0f;
            if (token == 1u) expected = float(d + 1u);
            if (token == 3u) expected = float(embed_dim + d + 1u);
            TEST(approx_eq(float(grad_embeddings.cpu()[token * embed_dim + d]), expected));
        }
    }
}

void test_position_add_and_backward() {
    printf("test_position_add_and_backward()\n");

    const uint32_t B = 2u;
    const uint32_t N = 2u;
    const uint32_t D = 3u;

    Tensor input({B, N, D});
    Tensor pos_emb({N, D});
    Tensor output({B, N, D});
    Tensor grad_out({B, N, D});
    Tensor grad_input({B, N, D});
    Tensor grad_pos({N, D});

    input.fill(float16_t(0.0f));
    float16_t* pp = pos_emb.cpu();
    for (uint32_t i = 0; i < N * D; ++i) {
        pp[i] = float16_t(float(i + 1));
    }
    pos_emb.cpu_upload();

    evk::ai::position_add(input, pos_emb, output, B, N, D);
    output.cpu_download();
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t i = 0; i < N * D; ++i) {
            TEST(approx_eq(float(output.cpu()[b * N * D + i]), float(pp[i])));
        }
    }

    grad_out.fill(float16_t(1.0f));
    grad_input.fill(float16_t(0.0f));
    grad_pos.fill(float16_t(0.0f));
    evk::ai::position_add_backward(grad_out, grad_input, grad_pos, B, N, D);

    grad_input.cpu_download();
    grad_pos.cpu_download();
    for (uint32_t i = 0; i < B * N * D; ++i) {
        TEST(approx_eq(float(grad_input.cpu()[i]), 1.0f));
    }
    for (uint32_t i = 0; i < N * D; ++i) {
        TEST(approx_eq(float(grad_pos.cpu()[i]), float(B)));
    }
}

void test_rope_and_backward() {
    printf("test_rope_and_backward()\n");

    const uint32_t B = 1u;
    const uint32_t N = 2u;
    const uint32_t D = 4u;
    const float base = 10000.0f;

    Tensor input({B, N, D});
    Tensor output({B, N, D});
    Tensor grad_out({B, N, D});
    Tensor grad_input({B, N, D});

    float host_input[B * N * D] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    float16_t* ip = input.cpu();
    for (uint32_t i = 0; i < B * N * D; ++i) {
        ip[i] = float16_t(host_input[i]);
    }
    input.cpu_upload();

    evk::ai::rope(input, output, B, N, D, base);
    output.cpu_download();

    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t pair = 0; pair < D / 2u; ++pair) {
            uint32_t base_idx = t * D + pair * 2u;
            float x0 = host_input[base_idx + 0u];
            float x1 = host_input[base_idx + 1u];
            float inv_freq = std::pow(base, -2.0f * float(pair) / float(D));
            float angle = float(t) * inv_freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            TEST(approx_eq(float(output.cpu()[base_idx + 0u]), x0 * c - x1 * s, 3e-2f));
            TEST(approx_eq(float(output.cpu()[base_idx + 1u]), x0 * s + x1 * c, 3e-2f));
        }
    }

    float host_grad[B * N * D] = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
    };
    float16_t* gop = grad_out.cpu();
    for (uint32_t i = 0; i < B * N * D; ++i) {
        gop[i] = float16_t(host_grad[i]);
    }
    grad_out.cpu_upload();
    grad_input.fill(float16_t(0.0f));

    evk::ai::rope_backward(grad_out, grad_input, B, N, D, base);
    grad_input.cpu_download();

    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t pair = 0; pair < D / 2u; ++pair) {
            uint32_t base_idx = t * D + pair * 2u;
            float g0 = host_grad[base_idx + 0u];
            float g1 = host_grad[base_idx + 1u];
            float inv_freq = std::pow(base, -2.0f * float(pair) / float(D));
            float angle = float(t) * inv_freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            TEST(approx_eq(float(grad_input.cpu()[base_idx + 0u]), g0 * c + g1 * s, 3e-2f));
            TEST(approx_eq(float(grad_input.cpu()[base_idx + 1u]), -g0 * s + g1 * c, 3e-2f));
        }
    }
}

void test_sum_batch() {
    printf("test_sum_batch()\n");

    Tensor input({2, 3});
    Tensor output({3});

    float16_t* ip = input.cpu();
    ip[0] = float16_t(1.0f);
    ip[1] = float16_t(2.0f);
    ip[2] = float16_t(3.0f);
    ip[3] = float16_t(4.0f);
    ip[4] = float16_t(5.0f);
    ip[5] = float16_t(6.0f);
    input.cpu_upload();

    output.fill(float16_t(1.0f));
    evk::ai::sum_batch(input, output, 2u, 3u);
    output.cpu_download();

    TEST(approx_eq(float(output.cpu()[0]), 6.0f));
    TEST(approx_eq(float(output.cpu()[1]), 8.0f));
    TEST(approx_eq(float(output.cpu()[2]), 10.0f));
}

void test_rms_norm_and_backward() {
    printf("test_rms_norm_and_backward()\n");

    Tensor input({2, 4});
    Tensor output({2, 4});
    Tensor grad_out({2, 4});
    Tensor grad_input({2, 4});

    float host_input[8] = {
        3.0f, 4.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
    };
    float16_t* ip = input.cpu();
    for (uint32_t i = 0; i < 8; ++i) {
        ip[i] = float16_t(host_input[i]);
    }
    input.cpu_upload();

    const float eps = 1e-4f;
    evk::ai::rms_norm(input, output, eps);
    output.cpu_download();

    for (uint32_t row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < 4; ++d) {
            float v = host_input[row * 4 + d];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / 4.0f + eps);
        for (uint32_t d = 0; d < 4; ++d) {
            float expected = host_input[row * 4 + d] * inv_rms;
            TEST(approx_eq(float(output.cpu()[row * 4 + d]), expected, 3e-2f));
        }
    }

    grad_out.fill(float16_t(1.0f));
    grad_input.fill(float16_t(0.0f));
    evk::ai::rms_norm_backward(input, grad_out, grad_input, eps);
    grad_input.cpu_download();

    for (uint32_t row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < 4; ++d) {
            float v = host_input[row * 4 + d];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / 4.0f + eps);
        float inv_rms = 1.0f / rms;
        float dot_gy = 0.0f;
        for (uint32_t d = 0; d < 4; ++d) {
            dot_gy += host_input[row * 4 + d] * inv_rms;
        }
        float scale = dot_gy / 4.0f;
        for (uint32_t d = 0; d < 4; ++d) {
            float x = host_input[row * 4 + d];
            float expected = inv_rms - x * inv_rms * inv_rms * scale;
            TEST(approx_eq(float(grad_input.cpu()[row * 4 + d]), expected, 4e-2f));
        }
    }
}

void test_greedy_sample() {
    printf("test_greedy_sample()\n");

    Tensor logits({2, 2, 5});
    Tensor out_tokens({2});

    float16_t* lp = logits.cpu();
    for (uint32_t i = 0; i < logits.shape.count(); ++i) {
        lp[i] = float16_t(-10.0f);
    }
    // Batch 0, position 1 -> token 3 wins in [1, 3]
    lp[(0u * 2u + 1u) * 5u + 1u] = float16_t(0.5f);
    lp[(0u * 2u + 1u) * 5u + 2u] = float16_t(2.5f);
    lp[(0u * 2u + 1u) * 5u + 3u] = float16_t(1.5f);
    // Batch 1, position 1 -> token 1 wins in [1, 3]
    lp[(1u * 2u + 1u) * 5u + 1u] = float16_t(3.0f);
    lp[(1u * 2u + 1u) * 5u + 2u] = float16_t(2.0f);
    lp[(1u * 2u + 1u) * 5u + 3u] = float16_t(1.0f);
    logits.cpu_upload();

    evk::ai::greedy_sample(logits, out_tokens, 1u, 1u, 3u);
    out_tokens.cpu_download();

    TEST(out_tokens.cpu()[0].value == 2u);
    TEST(out_tokens.cpu()[1].value == 1u);
}

void test_flash_attention_backward_scratch() {
    printf("test_flash_attention_backward_scratch()\n");

    const uint32_t B = 1u;
    const uint32_t N = 16u;
    const uint32_t Dh = 32u;
    const uint32_t H = 1u;
    const uint32_t D = H * Dh;

    Tensor q({B, N, D});
    Tensor k({B, N, Dh});
    Tensor v({B, N, Dh});
    Tensor o({B, N, D});
    Tensor dO({B, N, D});
    Tensor dQ({B, N, D});
    Tensor dK({B, N, Dh});
    Tensor dV({B, N, Dh});

    q.fill(float16_t(0.0f));
    k.fill(float16_t(0.0f));
    v.fill(float16_t(0.0f));
    o.fill(float16_t(0.0f));
    dQ.fill(float16_t(0.0f));
    dK.fill(float16_t(0.0f));
    dV.fill(float16_t(0.0f));

    float16_t* dop = dO.cpu();
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            dop[i * D + d] = float16_t(float(d + 1u));
        }
    }
    dO.cpu_upload();

    // Intentionally call backward without a prior forward pass.
    evk::ai::flash_attention_bwd(q, k, v, o, dO, dQ, dK, dV);

    dQ.cpu_download();
    dK.cpu_download();
    dV.cpu_download();

    bool ok_dq = true;
    bool ok_dk = true;
    bool ok_dv = true;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            if (!approx_eq(float(dQ.cpu()[i * D + d]), 0.0f, 5e-2f)) {
                ok_dq = false;
            }
        }
    }
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t d = 0; d < Dh; ++d) {
            if (!approx_eq(float(dK.cpu()[j * Dh + d]), 0.0f, 5e-2f)) {
                ok_dk = false;
            }
            if (!approx_eq(float(dV.cpu()[j * Dh + d]), float(d + 1u), 8e-2f)) {
                ok_dv = false;
            }
        }
    }
    TEST(ok_dq);
    TEST(ok_dk);
    TEST(ok_dv);
}

void run_ai_kernel_tests() {
    test_add();
    test_matmul();
    test_matmul_broadcast();
    test_mse_loss();
    test_cross_entropy_loss();
    test_sgd();
    test_adam();
    test_adam_fp16_matmul_convergence_vs_cpu_fp32();
    test_softmax();
    test_relu();
    test_gelu();
    test_scale_zero();
    test_embed_and_backward();
    test_position_add_and_backward();
    test_rope_and_backward();
    test_sum_batch();
    test_rms_norm_and_backward();
    test_greedy_sample();
    test_flash_attention_backward_scratch();
}
