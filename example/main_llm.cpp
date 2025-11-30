#include <iostream>
#include <evk_ai.h>

#include "bmp.h"

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

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

// 2D FFT mixing for FNet (CPU implementation)
// Applies 2D DFT along sequence and hidden dimensions, takes real part
void fft_mix_forward(Tensor& input, Tensor& output, uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim) {
    input.cpu_download();
    float16_t* inp = input.cpu();
    float16_t* outp = output.cpu();
    
    const float PI = 3.14159265358979323846f;
    
    // For each batch
    for (uint32_t b = 0; b < batch_size; ++b) {
        // Apply 2D DFT: first along sequence dimension, then along hidden dimension
        // We'll use a simplified real-valued approximation for efficiency
        // FNet paper: y = Real(FFT_seq(FFT_hidden(x)))
        
        // Temp buffer for intermediate FFT
        std::vector<float> temp_real(seq_len * embed_dim, 0.0f);
        std::vector<float> temp_imag(seq_len * embed_dim, 0.0f);
        
        // Step 1: FFT along hidden dimension (for each position)
        for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t k = 0; k < embed_dim; ++k) {
                float sum_real = 0.0f;
                float sum_imag = 0.0f;
                for (uint32_t n = 0; n < embed_dim; ++n) {
                    uint32_t idx = b * seq_len * embed_dim + t * embed_dim + n;
                    float x = float(inp[idx]);
                    float angle = -2.0f * PI * float(k) * float(n) / float(embed_dim);
                    sum_real += x * cosf(angle);
                    sum_imag += x * sinf(angle);
                }
                temp_real[t * embed_dim + k] = sum_real;
                temp_imag[t * embed_dim + k] = sum_imag;
            }
        }
        
        // Step 2: FFT along sequence dimension (for each hidden dim)
        for (uint32_t d = 0; d < embed_dim; ++d) {
            for (uint32_t k = 0; k < seq_len; ++k) {
                float sum_real = 0.0f;
                float sum_imag = 0.0f;
                for (uint32_t n = 0; n < seq_len; ++n) {
                    float xr = temp_real[n * embed_dim + d];
                    float xi = temp_imag[n * embed_dim + d];
                    float angle = -2.0f * PI * float(k) * float(n) / float(seq_len);
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    // Complex multiplication: (xr + i*xi) * (cos_a + i*sin_a)
                    sum_real += xr * cos_a - xi * sin_a;
                    sum_imag += xr * sin_a + xi * cos_a;
                }
                // Take real part only (as per FNet paper)
                uint32_t out_idx = b * seq_len * embed_dim + k * embed_dim + d;
                outp[out_idx] = float16_t(sum_real / sqrtf(float(seq_len * embed_dim)));
            }
        }
    }
    output.cpu_upload();
}

// FFT mixing backward pass
void fft_mix_backward(Tensor& grad_out, Tensor& grad_in, uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim) {
    grad_out.cpu_download();
    grad_in.cpu_download();
    float16_t* go = grad_out.cpu();
    float16_t* gi = grad_in.cpu();
    
    const float PI = 3.14159265358979323846f;
    float scale = 1.0f / sqrtf(float(seq_len * embed_dim));
    
    // The backward of real(FFT(x)) is the real part of conjugate(FFT(grad))
    // Simplified: we treat this as a linear operation and backprop accordingly
    for (uint32_t b = 0; b < batch_size; ++b) {
        // Apply inverse 2D DFT (conjugate of forward)
        std::vector<float> temp_real(seq_len * embed_dim, 0.0f);
        std::vector<float> temp_imag(seq_len * embed_dim, 0.0f);
        
        // Step 1: Inverse FFT along sequence dimension
        for (uint32_t d = 0; d < embed_dim; ++d) {
            for (uint32_t n = 0; n < seq_len; ++n) {
                float sum_real = 0.0f;
                for (uint32_t k = 0; k < seq_len; ++k) {
                    uint32_t idx = b * seq_len * embed_dim + k * embed_dim + d;
                    float g = float(go[idx]) * scale;
                    float angle = 2.0f * PI * float(k) * float(n) / float(seq_len);
                    sum_real += g * cosf(angle);
                }
                temp_real[n * embed_dim + d] = sum_real;
            }
        }
        
        // Step 2: Inverse FFT along hidden dimension
        for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t n = 0; n < embed_dim; ++n) {
                float sum_real = 0.0f;
                for (uint32_t k = 0; k < embed_dim; ++k) {
                    float g = temp_real[t * embed_dim + k];
                    float angle = 2.0f * PI * float(k) * float(n) / float(embed_dim);
                    sum_real += g * cosf(angle);
                }
                uint32_t out_idx = b * seq_len * embed_dim + t * embed_dim + n;
                gi[out_idx] = float16_t(float(gi[out_idx]) + sum_real);
            }
        }
    }
    grad_in.cpu_upload();
}

void test_next_token_prediction_fnet() {
    printf("test_next_token_prediction_fnet()\n");
    
    // ========================================
    // 1. Vocabulary and tokenization (same as attention version)
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
    // 2. Lorem ipsum training samples (same as attention version)
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
    // 3. Model hyperparameters
    // ========================================
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    
    printf("  FNet Model: vocab=%u, seq=%u, embed=%u, hidden=%u\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    printf("  Using Fourier Transform mixing instead of attention\n");
    
    // ========================================
    // 4. Build FNet computation graph
    // ========================================
    Graph model;
    
    // Input tokens (filled each batch) - indices as uint16
    Tensor& input_tokens = model.tensor({BATCH_SIZE * SEQ_LEN});
    Tensor& targets = model.tensor({BATCH_SIZE * SEQ_LEN});
    
    // Embedding parameters (learnable)
    Tensor& token_emb = model.tensor({VOCAB_SIZE, EMBED_DIM}, true);
    Tensor& pos_emb = model.tensor({SEQ_LEN, EMBED_DIM}, true);
    
    // FNet layers: FFT mixing + MLP
    Tensor& w1 = model.tensor({EMBED_DIM, HIDDEN_DIM}, true);
    Tensor& w2 = model.tensor({HIDDEN_DIM, EMBED_DIM}, true);
    Tensor& w_out = model.tensor({EMBED_DIM, VOCAB_SIZE}, true);
    
    // Token embedding lookup
    Tensor& embedded = model.embed(token_emb, input_tokens);
    
    // Add positional embeddings (broadcast across batch)
    Tensor& input_with_pos = model.add_position_embedding(embedded, pos_emb, BATCH_SIZE, SEQ_LEN);
    
    // FFT mixing layer output (created manually)
    Tensor& fft_mixed = model.tensor({BATCH_SIZE * SEQ_LEN, EMBED_DIM});
    
    // Setup FFT forward and backward functions
    fft_mixed.forward_fn = [&input_with_pos, &fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM]() {
        fft_mix_forward(input_with_pos, fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM);
    };
    
    fft_mixed.backward_fn = [&input_with_pos, &fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM]() {
        fft_mix_backward(fft_mixed.grad(), input_with_pos.grad(), BATCH_SIZE, SEQ_LEN, EMBED_DIM);
    };
    
    // Residual connection with FFT output
    Tensor& fft_residual = model.add(input_with_pos, fft_mixed);
    
    // Feedforward network (same as attention version)
    Tensor& hidden = model.matmul(fft_residual, w1);
    Tensor& hidden_relu = model.relu(hidden);
    Tensor& hidden_proj = model.matmul(hidden_relu, w2);
    Tensor& ff_residual = model.add(fft_residual, hidden_proj);
    Tensor& logits = model.matmul(ff_residual, w_out);
    Tensor& loss = model.cross_entropy_loss(logits, targets);
    
    // ========================================
    // 5. Initialize weights
    // ========================================
    srand(42); // Same seed for fair comparison
    float scale = 0.1f / sqrtf(float(EMBED_DIM));
    token_emb.random_init(0.1f);
    pos_emb.random_init(0.1f);
    w1.random_init(scale);
    w2.random_init(scale);
    w_out.random_init(scale);
    
    // ========================================
    // 6. Training loop
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
            // Prepare batch data
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
            
            // Forward + Backward pass
            model.eval(true);
            evk::Sync();
            
            loss.cpu_download();
            float loss_val = float(loss.cpu()[0]);
            epoch_loss += loss_val;
            num_batches++;
            
            if (epoch == 0 && batch_start == 0) {
                initial_loss = loss_val;
            }
            
            // Optimizer step
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
    // 7. Autoregressive generation
    // ========================================
    printf("\n  FNet Autoregressive generation:\n");
    
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
            uint16_t* inp = (uint16_t*)input_tokens.cpu();
            memset(inp, 0, BATCH_SIZE * SEQ_LEN * sizeof(uint16_t));
            for (uint32_t i = 0; i < cur_len; ++i) {
                inp[i] = tokens[i];
            }
            input_tokens.cpu_upload();
            
            model.eval(false);
            evk::Sync();
            
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
    printf("\n  FNet Validation:\n");
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initial_loss, last_loss);
    
    TEST(last_loss < initial_loss);
    TEST(last_loss < 2.5f);
}

// Returns final loss for comparison
float run_next_token_prediction_attention() {
    printf("\n=== Attention-based Transformer ===\n");
    
    const char* VOCAB = " abcdefghijklmnopqrstuvwxyz.,\n";
    const uint32_t VOCAB_SIZE = 32;
    const uint32_t ACTUAL_VOCAB = 30;
    
    auto char_to_token = [&](char c) -> uint16_t {
        for (uint32_t i = 0; VOCAB[i]; ++i) {
            if (VOCAB[i] == c) return uint16_t(i);
        }
        return 0;
    };
    
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
    
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    
    printf("  MLP Model (baseline): vocab=%u, seq=%u, embed=%u, hidden=%u\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    
    Graph model;
    
    Tensor& input_tokens = model.tensor({BATCH_SIZE * SEQ_LEN});
    Tensor& targets = model.tensor({BATCH_SIZE * SEQ_LEN});
    
    Tensor& token_emb = model.tensor({VOCAB_SIZE, EMBED_DIM}, true);
    Tensor& pos_emb = model.tensor({SEQ_LEN, EMBED_DIM}, true);
    
    Tensor& w1 = model.tensor({EMBED_DIM, HIDDEN_DIM}, true);
    Tensor& w2 = model.tensor({HIDDEN_DIM, EMBED_DIM}, true);
    Tensor& w_out = model.tensor({EMBED_DIM, VOCAB_SIZE}, true);
    
    Tensor& embedded = model.embed(token_emb, input_tokens);
    Tensor& input_with_pos = model.add_position_embedding(embedded, pos_emb, BATCH_SIZE, SEQ_LEN);
    
    Tensor& hidden = model.matmul(input_with_pos, w1);
    Tensor& hidden_relu = model.relu(hidden);
    Tensor& hidden_proj = model.matmul(hidden_relu, w2);
    Tensor& residual = model.add(input_with_pos, hidden_proj);
    Tensor& logits = model.matmul(residual, w_out);
    Tensor& loss = model.cross_entropy_loss(logits, targets);
    
    srand(42);
    float scale = 0.1f / sqrtf(float(EMBED_DIM));
    token_emb.random_init(0.1f);
    pos_emb.random_init(0.1f);
    w1.random_init(scale);
    w2.random_init(scale);
    w_out.random_init(scale);
    
    const int EPOCHS = 200;
    const float LR = 0.01f;
    
    printf("  Training for %d epochs...\n", EPOCHS);
    
    float initial_loss = 0.0f;
    float last_loss = 1e9f;
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (uint32_t batch_start = 0; batch_start + BATCH_SIZE <= NUM_SAMPLES; batch_start += BATCH_SIZE) {
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
            
            model.eval(true);
            evk::Sync();
            
            loss.cpu_download();
            float loss_val = float(loss.cpu()[0]);
            epoch_loss += loss_val;
            num_batches++;
            
            if (epoch == 0 && batch_start == 0) {
                initial_loss = loss_val;
            }
            
            model.step_adam(-LR);
            evk::Sync();
        }
        
        float avg_loss = epoch_loss / float(num_batches);
        
        if (epoch % 40 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, avg_loss);
        }
        
        last_loss = avg_loss;
    }
    
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initial_loss, last_loss);
    return last_loss;
}

// Returns final loss for comparison
float run_next_token_prediction_fnet() {
    printf("\n=== FNet (Fourier Transform) ===\n");
    
    const char* VOCAB = " abcdefghijklmnopqrstuvwxyz.,\n";
    const uint32_t VOCAB_SIZE = 32;
    const uint32_t ACTUAL_VOCAB = 30;
    
    auto char_to_token = [&](char c) -> uint16_t {
        for (uint32_t i = 0; VOCAB[i]; ++i) {
            if (VOCAB[i] == c) return uint16_t(i);
        }
        return 0;
    };
    
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
    
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    
    printf("  FNet Model: vocab=%u, seq=%u, embed=%u, hidden=%u\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    printf("  Using 2D Fourier Transform mixing\n");
    
    Graph model;
    
    Tensor& input_tokens = model.tensor({BATCH_SIZE * SEQ_LEN});
    Tensor& targets = model.tensor({BATCH_SIZE * SEQ_LEN});
    
    Tensor& token_emb = model.tensor({VOCAB_SIZE, EMBED_DIM}, true);
    Tensor& pos_emb = model.tensor({SEQ_LEN, EMBED_DIM}, true);
    
    Tensor& w1 = model.tensor({EMBED_DIM, HIDDEN_DIM}, true);
    Tensor& w2 = model.tensor({HIDDEN_DIM, EMBED_DIM}, true);
    Tensor& w_out = model.tensor({EMBED_DIM, VOCAB_SIZE}, true);
    
    Tensor& embedded = model.embed(token_emb, input_tokens);
    Tensor& input_with_pos = model.add_position_embedding(embedded, pos_emb, BATCH_SIZE, SEQ_LEN);
    
    // FFT mixing layer
    Tensor& fft_mixed = model.tensor({BATCH_SIZE * SEQ_LEN, EMBED_DIM});
    
    fft_mixed.forward_fn = [&input_with_pos, &fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM]() {
        fft_mix_forward(input_with_pos, fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM);
    };
    
    fft_mixed.backward_fn = [&input_with_pos, &fft_mixed, BATCH_SIZE, SEQ_LEN, EMBED_DIM]() {
        fft_mix_backward(fft_mixed.grad(), input_with_pos.grad(), BATCH_SIZE, SEQ_LEN, EMBED_DIM);
    };
    
    Tensor& fft_residual = model.add(input_with_pos, fft_mixed);
    
    Tensor& hidden = model.matmul(fft_residual, w1);
    Tensor& hidden_relu = model.relu(hidden);
    Tensor& hidden_proj = model.matmul(hidden_relu, w2);
    Tensor& ff_residual = model.add(fft_residual, hidden_proj);
    Tensor& logits = model.matmul(ff_residual, w_out);
    Tensor& loss = model.cross_entropy_loss(logits, targets);
    
    srand(42);
    float scale = 0.1f / sqrtf(float(EMBED_DIM));
    token_emb.random_init(0.1f);
    pos_emb.random_init(0.1f);
    w1.random_init(scale);
    w2.random_init(scale);
    w_out.random_init(scale);
    
    const int EPOCHS = 200;
    const float LR = 0.01f;
    
    printf("  Training for %d epochs...\n", EPOCHS);
    
    float initial_loss = 0.0f;
    float last_loss = 1e9f;
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (uint32_t batch_start = 0; batch_start + BATCH_SIZE <= NUM_SAMPLES; batch_start += BATCH_SIZE) {
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
            
            model.eval(true);
            evk::Sync();
            
            loss.cpu_download();
            float loss_val = float(loss.cpu()[0]);
            epoch_loss += loss_val;
            num_batches++;
            
            if (epoch == 0 && batch_start == 0) {
                initial_loss = loss_val;
            }
            
            model.step_adam(-LR);
            evk::Sync();
        }
        
        float avg_loss = epoch_loss / float(num_batches);
        
        if (epoch % 40 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, avg_loss);
        }
        
        last_loss = avg_loss;
    }
    
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initial_loss, last_loss);
    return last_loss;
}

void compare_attention_vs_fnet() {
    printf("\n========================================\n");
    printf("Comparing Attention vs FNet for Next Token Prediction\n");
    printf("========================================\n");
    
    float attention_loss = run_next_token_prediction_attention();
    float fnet_loss = run_next_token_prediction_fnet();
    
    printf("\n========================================\n");
    printf("COMPARISON RESULTS:\n");
    printf("========================================\n");
    printf("  MLP (baseline) final loss:  %.4f\n", attention_loss);
    printf("  FNet final loss:            %.4f\n", fnet_loss);
    printf("  Difference:                 %.4f\n", attention_loss - fnet_loss);
    
    if (fnet_loss < attention_loss) {
        printf("  -> FNet achieves LOWER loss (better)\n");
    } else if (fnet_loss > attention_loss) {
        printf("  -> MLP achieves LOWER loss (better)\n");
    } else {
        printf("  -> Both achieve similar loss\n");
    }
    printf("========================================\n");
    
    TEST(attention_loss < 2.5f);
    TEST(fnet_loss < 2.5f);
}

void main_llm() {
    printf("=== main_llm ===\n");
    // test_next_token_prediction();
    // test_next_token_prediction_fnet();
    compare_attention_vs_fnet();
}
