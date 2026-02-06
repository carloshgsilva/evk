#include <iostream>
#include <algorithm>
#include <evk_ai.h>

#include "bmp.h"

// ============================================================================
// Transformer Model
// ============================================================================
#if 0

// Attention block: Query-Only attention + FFN with residuals
struct AttentionBlock {
    // Weight tensors (owned by graph)
    Tensor* w_q = nullptr;  // Query projection
    Tensor* w_k = nullptr;  // Key projection (from base embedding)
    Tensor* w_v = nullptr;  // Value projection (from base embedding)
    Tensor* w_o = nullptr;  // Output projection
    Tensor* w1 = nullptr;   // FFN first layer
    Tensor* w2 = nullptr;   // FFN second layer
    
    // Dimensions
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t layer_idx;
    uint32_t total_layers;
    
    AttentionBlock() : embed_dim(0), hidden_dim(0), layer_idx(0), total_layers(1) {}
    
    // Initialize weights in the given graph
    void init(Graph& graph, uint32_t embed_dim_, uint32_t hidden_dim_, uint32_t layer_idx_ = 0, uint32_t total_layers_ = 1) {
        embed_dim = embed_dim_;
        hidden_dim = hidden_dim_;
        layer_idx = layer_idx_;
        total_layers = total_layers_;
        
        w_q = &graph.tensor({embed_dim, embed_dim}, true);
        w_k = &graph.tensor({embed_dim, embed_dim}, true);
        w_v = &graph.tensor({embed_dim, embed_dim}, true);
        w_o = &graph.tensor({embed_dim, embed_dim}, true);
        w1 = &graph.tensor({embed_dim, hidden_dim}, true);
        w2 = &graph.tensor({hidden_dim, embed_dim}, true);
    }
    
    // Initialize weights with random values using proper scaling for deep networks
    // Uses a combination of He initialization with residual scaling
    void init_weights(float base_scale, bool use_residual_scaling = true) {
        // Attention projections are more stable with slightly smaller init than pure He,
        // especially in fp16 and with Adam at relatively high LR.
        // Use Xavier-like scaling for Q/K/V/O.
        float attn_in_scale = base_scale * (1.0f / sqrtf(float(embed_dim)));

        // FFN uses ReLU, so He init is appropriate.
        float w1_scale = base_scale * sqrtf(2.0f / float(embed_dim));
        
        // For the last projection in residual path, scale down to prevent growth
        // This is the "residual scaling" technique from GPT-2 / muP
        float residual_scale = use_residual_scaling ? (1.0f / sqrtf(float(2 * total_layers))) : 1.0f;
        float w_o_scale = attn_in_scale * residual_scale;
        float w2_scale = base_scale * sqrtf(2.0f / float(hidden_dim)) * residual_scale;
        
        w_q->random_init(attn_in_scale);
        w_k->random_init(attn_in_scale);
        w_v->random_init(attn_in_scale);
        w_o->random_init(w_o_scale);
        w1->random_init(w1_scale);
        w2->random_init(w2_scale);
    }
    
    // Forward pass through this block
    // input:       (B, N, embed_dim) current layer residual stream
    // kv_base_norm:(B, N, embed_dim) RMSNorm(first embedding) (fixed across layers)
    // Returns:     (B, N, embed_dim) output with residual connections
    Tensor& forward(Graph& graph, Tensor& input, Tensor& kv_base_norm, float residual_scale = 1.0f) {
        // Pre-norm Transformer block variant:
        //   Attention uses:
        //     Q from current layer (RMSNorm(input) @ Wq)
        //     K,V from the *first* embedding (kv_base_norm)
        //   y = x + scale * Attn(Q(x), K(x0), V(x0))
        //   z = y + scale * FFN(RMSNorm(y))

        // 1) Attention block
        // Q from current residual stream (keeps stability similar to pre-norm)
        Tensor& q_in = graph.rms_norm(input);
        Tensor& q = graph.matmul(q_in, *w_q);

        // K/V are projections of the fixed (bottom) embedding, not from intermediate residuals.
        Tensor& k = graph.matmul(kv_base_norm, *w_k);
        Tensor& v = graph.matmul(kv_base_norm, *w_v);
        Tensor& attn_out = graph.causal_attention(q, k, v);

        // Output projection
        Tensor& attn_proj = graph.matmul(attn_out, *w_o);
        
        // Optional: scale the attention output before residual (helps with deep networks)
        Tensor* attn_scaled = &attn_proj;
        if (residual_scale != 1.0f) {
            attn_scaled = &graph.scale(attn_proj, residual_scale);
        }
        Tensor& attn_residual = graph.add(input, *attn_scaled);
        
        // 2) Pre-norm FFN block
        Tensor& norm2 = graph.rms_norm(attn_residual);
        Tensor& hidden = graph.matmul(norm2, *w1);
        Tensor& hidden_relu = graph.relu(hidden);
        Tensor& hidden_proj = graph.matmul(hidden_relu, *w2);
        
        // Optional: scale the FFN output before residual
        Tensor* ffn_scaled = &hidden_proj;
        if (residual_scale != 1.0f) {
            ffn_scaled = &graph.scale(hidden_proj, residual_scale);
        }
        Tensor& output = graph.add(attn_residual, *ffn_scaled);
        
        return output;
    }
};

// Query-Only Attention Transformer with N layers
// Takes uint16_t tokens as input directly
struct Transformer {
    // Hyperparameters
    uint32_t vocab_size;
    uint32_t seq_len;
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t batch_size;
    uint32_t num_layers;
    
    // Training graph
    Graph model;
    Tensor* input_tokens = nullptr;
    Tensor* targets = nullptr;
    Tensor* loss = nullptr;
    
    // Weight tensors (owned by model graph)
    Tensor* token_emb = nullptr;
    Tensor* pos_emb = nullptr;
    Tensor* w_out = nullptr;

    
    
    // Attention blocks (N layers)
    std::vector<AttentionBlock> blocks;
    
    // Inference graph (separate for generation)
    Graph inference;
    Tensor* inf_input = nullptr;
    Tensor* inf_logits = nullptr;
    
    // Inference weight tensors (copies from training)
    Tensor* inf_token_emb = nullptr;
    Tensor* inf_pos_emb = nullptr;
    Tensor* inf_w_out = nullptr;
    
    // Inference attention blocks
    std::vector<AttentionBlock> inf_blocks;
    
    Transformer(uint32_t vocab_size, uint32_t seq_len, uint32_t embed_dim, 
                uint32_t hidden_dim, uint32_t batch_size, uint32_t num_layers = 1)
        : vocab_size(vocab_size), seq_len(seq_len), embed_dim(embed_dim),
                    hidden_dim(hidden_dim), batch_size(batch_size), num_layers(num_layers)
    {
        build_training_graph();
        build_inference_graph();
    }

    uint64_t trainable_param_count() const {
        uint64_t total = 0;
        if (token_emb) total += token_emb->shape.count();
        if (pos_emb) total += pos_emb->shape.count();
        if (w_out) total += w_out->shape.count();
        for (const auto& b : blocks) {
            if (b.w_q) total += b.w_q->shape.count();
            if (b.w_k) total += b.w_k->shape.count();
            if (b.w_v) total += b.w_v->shape.count();
            if (b.w_o) total += b.w_o->shape.count();
            if (b.w1) total += b.w1->shape.count();
            if (b.w2) total += b.w2->shape.count();
        }
        return total;
    }
    
    void build_training_graph() {
        // Input tensors
        input_tokens = &model.tensor({batch_size, seq_len});
        targets = &model.tensor({batch_size, seq_len});
        
        // Learnable parameters
        token_emb = &model.tensor({vocab_size, embed_dim}, true);
        pos_emb = &model.tensor({seq_len, embed_dim}, true);
        w_out = &model.tensor({embed_dim, vocab_size}, true);
        
        // Initialize attention blocks with layer indices for proper scaling
        blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init(model, embed_dim, hidden_dim, i, num_layers);
        }
        
        // Forward pass
        Tensor& embedded = model.embed(*token_emb, *input_tokens);
        Tensor& input_with_pos = model.add_position_embedding(embedded, *pos_emb, batch_size, seq_len);

        // Attention K/V source: always the first embedding (token+pos) for all layers.
        // We RMS-normalize it once and reuse across all attention blocks.
        Tensor& kv_base_norm = model.rms_norm(input_with_pos);
        
        // Residual output scaling.
        // For the base-KV attention variant, scaling residual contributions down by depth
        // can make optimization harder, so we disable it by default.
        constexpr bool USE_DEPTH_RESIDUAL_SCALE = false;
        float residual_scale = 1.0f;
        if (USE_DEPTH_RESIDUAL_SCALE && num_layers > 2) {
            residual_scale = 1.0f / sqrtf(float(num_layers));
        }
        
        // Pass through N attention blocks
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &blocks[i].forward(model, *x, kv_base_norm, residual_scale);
        }

        // Output projection and loss
        Tensor& logits = model.matmul(*x, *w_out);
        loss = &model.cross_entropy_loss(logits, *targets);
    }
    
    void build_inference_graph() {
        // Single-batch inference
        inf_input = &inference.tensor({1, seq_len});
        
        // Weight tensors (non-trainable, will be copied from training)
        inf_token_emb = &inference.tensor({vocab_size, embed_dim});
        inf_pos_emb = &inference.tensor({seq_len, embed_dim});
        inf_w_out = &inference.tensor({embed_dim, vocab_size});
        
        // Initialize inference attention blocks
        inf_blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            inf_blocks[i].init(inference, embed_dim, hidden_dim, i, num_layers);
        }
        
        // Compute same residual scale as training
        constexpr bool USE_DEPTH_RESIDUAL_SCALE = false;
        float residual_scale = 1.0f;
        if (USE_DEPTH_RESIDUAL_SCALE && num_layers > 2) {
            residual_scale = 1.0f / sqrtf(float(num_layers));
        }
        
        // Forward pass (same architecture as training)
        Tensor& embedded = inference.embed(*inf_token_emb, *inf_input);
        Tensor& input_with_pos = inference.add_position_embedding(embedded, *inf_pos_emb, 1, seq_len);

        // Attention K/V source: always the first embedding (token+pos) for all layers.
        Tensor& kv_base_norm = inference.rms_norm(input_with_pos);
        
        // Pass through N attention blocks (use same residual_scale as training)
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &inf_blocks[i].forward(inference, *x, kv_base_norm, residual_scale);
        }

        inf_logits = &inference.matmul(*x, *inf_w_out);
    }
    
    void init_weights(uint32_t seed = 42) {
        srand(seed);
        
        // Proper initialization for deep transformers:
        // - Embeddings: use small values to keep activations bounded
        // - Weights: use He-like scaling with residual compensation
        float emb_scale = 0.02f;  // Small embedding init
        // If Hyperball is disabled, keep init a bit smaller to avoid early optimizer blow-ups.
        float base_scale = 0.25f;
        
        token_emb->random_init(emb_scale);
        pos_emb->random_init(emb_scale);
        
        // Initialize all attention block weights with proper scaling
        for (uint32_t i = 0; i < num_layers; ++i) {
            // With Hyperball we explicitly constrain matrix norms; residual-scaled init can
            // lock some residual-path projections (w_o, w2) to overly small norms.
            // So we disable residual scaling at init and let Hyperball control norms.
            blocks[i].init_weights(base_scale, false);
        }
        
        // Output projection: use smaller init to prevent large initial logits
        float w_out_scale = 0.02f / sqrtf(float(embed_dim));
        w_out->random_init(w_out_scale);
        
    }
    
    void copy_weights_to_inference() {
        auto& cmd = evk::ai::GetCmd();
        cmd.copy(token_emb->buffer, inf_token_emb->buffer, token_emb->shape.count() * sizeof(float16_t));
        cmd.copy(pos_emb->buffer, inf_pos_emb->buffer, pos_emb->shape.count() * sizeof(float16_t));
        cmd.copy(w_out->buffer, inf_w_out->buffer, w_out->shape.count() * sizeof(float16_t));
        
        // Copy all attention block weights
        for (uint32_t i = 0; i < num_layers; ++i) {
            cmd.copy(blocks[i].w_q->buffer, inf_blocks[i].w_q->buffer, blocks[i].w_q->shape.count() * sizeof(float16_t));
            cmd.copy(blocks[i].w_k->buffer, inf_blocks[i].w_k->buffer, blocks[i].w_k->shape.count() * sizeof(float16_t));
            cmd.copy(blocks[i].w_v->buffer, inf_blocks[i].w_v->buffer, blocks[i].w_v->shape.count() * sizeof(float16_t));
            cmd.copy(blocks[i].w_o->buffer, inf_blocks[i].w_o->buffer, blocks[i].w_o->shape.count() * sizeof(float16_t));
            cmd.copy(blocks[i].w1->buffer, inf_blocks[i].w1->buffer, blocks[i].w1->shape.count() * sizeof(float16_t));
            cmd.copy(blocks[i].w2->buffer, inf_blocks[i].w2->buffer, blocks[i].w2->shape.count() * sizeof(float16_t));
        }
        evk::ai::SubmitCmd(true);
    }
    
    // Train on a single batch with raw token arrays
    // input: (batch_size * seq_len) array of input tokens
    // target: (batch_size * seq_len) array of target tokens
    // Returns the loss value for this batch
    float train_batch(const uint16_t* input, const uint16_t* target, float learning_rate) {
        uint16_t* inp = (uint16_t*)input_tokens->cpu();
        uint16_t* tgt = (uint16_t*)targets->cpu();
        
        memcpy(inp, input, batch_size * seq_len * sizeof(uint16_t));
        memcpy(tgt, target, batch_size * seq_len * sizeof(uint16_t));
        
        input_tokens->cpu_upload(false);
        targets->cpu_upload(false);
        
        model.eval(true, false, false);
        
        loss->cpu_download(false);

        // Optimizer:
        // - Embeddings (token_emb, pos_emb): vanilla Adam
        // - All other projection matrices: Adam-Hyperball (fixed Frobenius norms)
        constexpr float BETA1 = 0.9f;
        constexpr float BETA2 = 0.98f;
        constexpr float EPS   = 5e-5f;

        for (auto* param : model.params) {
            assert(param->grad_tensor);
            evk::ai::adam(*param, param->grad(), model.adam_states[param],
                          learning_rate, BETA1, BETA2, EPS);
        }
        evk::ai::SubmitCmd(true);
        
        float loss_val = float(loss->cpu()[0]);

        // model.step(-learning_rate*0.1f);
        
        return loss_val;
    }
    
    // Generate tokens autoregressively from a prefix
    // prefix: array of prefix tokens
    // prefix_len: length of prefix
    // output: array to store generated tokens (must be at least prefix_len + max_new_tokens)
    // max_new_tokens: maximum number of new tokens to generate
    // Returns the total length of generated sequence (prefix + new tokens)
    uint32_t generate(const uint16_t* prefix, uint32_t prefix_len, uint16_t* output, int max_new_tokens) {
        // Ensure inference weights are up to date
        copy_weights_to_inference();
        
        // Copy prefix to output
        memcpy(output, prefix, prefix_len * sizeof(uint16_t));
        uint32_t cur_len = prefix_len;
        
        for (int g = 0; g < max_new_tokens && cur_len < seq_len; ++g) {
            // Fill input tokens
            uint16_t* inp = (uint16_t*)inf_input->cpu();
            for (uint32_t t = 0; t < seq_len; ++t) {
                inp[t] = (t < cur_len) ? output[t] : uint16_t(0);
            }
            inf_input->cpu_upload();
            
            // Forward pass (no backward)
            inference.eval(false);
            
            // Get logits for the last position
            inf_logits->cpu_download();
            float16_t* log_ptr = inf_logits->cpu();
            
            // Find argmax at position (cur_len - 1)
            uint32_t last_pos = cur_len - 1;
            float max_val = -1e9f;
            uint16_t max_idx = 0;
            for (uint32_t v = 0; v < vocab_size; ++v) {
                float val = float(log_ptr[last_pos * vocab_size + v]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = uint16_t(v);
                }
            }
            
            // Append generated token
            output[cur_len] = max_idx;
            cur_len++;
        }
        
        return cur_len;
    }
};
#endif

// Forward declaration
void run_circle_drifting();

#include <chrono>
void main_llm() {
    evk::ai::initialize();
    printf("=== main_llm ===\n");
    // run_next_token_prediction_attention();  // Comment out for now

    auto start = std::chrono::high_resolution_clock::now();
    // Discrete autoregressive circle-token prediction (legacy):
    // run_circle_detection(6);

    // New: single-step drifting over continuous circle parameters.
    run_circle_drifting();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("main_llm() took %.4f seconds\n", duration.count());
    evk::ai::shutdown();
}

// ============================================================================
// Circle Detection Transformer
// ============================================================================
// Takes N_POINTS 2D points as input and outputs N_MAX_PRIMS circle parameters
// Each circle is defined by (x, y, radius) with discrete/quantized coordinates
// Uses a set-to-sequence transformer architecture

struct CircleData {
    // Circle parameters: x, y, radius (discretized)
    uint16_t x;
    uint16_t y; 
    uint16_t r;
};

struct PointData {
    uint16_t x;
    uint16_t y;
};

// Generate synthetic training data: random circles with points sampled from them
struct CircleDataset {
    uint32_t grid_size;
    uint32_t min_radius;
    uint32_t max_radius;
    
    uint32_t n_points;      // Number of input points per sample
    uint32_t n_max_prims;   // Maximum number of circles to generate
    
    CircleDataset(uint32_t n_points_, uint32_t n_max_prims_, uint32_t grid_size_ = 32)
        : grid_size(grid_size_), n_points(n_points_), n_max_prims(n_max_prims_)
    {
        min_radius = grid_size / 8;
        max_radius = grid_size / 3;
    }
    
    // Generate a single sample: random circles + sampled points
    // Returns the number of circles generated (1 to n_max_prims)
    uint32_t generate_sample(CircleData* circles_out, PointData* points_out, uint32_t force_num_circles = 0) {
        uint32_t num_circles = force_num_circles ? force_num_circles : (1 + (rand() % n_max_prims));
        num_circles = (std::max)(1u, (std::min)(num_circles, n_max_prims));

        // NOTE:
        // For large num_circles, the original radius range (grid/8..grid/3) leads to heavy overlap
        // in a small grid, making exact recovery from quantized points extremely ambiguous.
        // To make the 5-circle case learnable, shrink radii as circle-count grows and encourage separation.
        uint32_t local_min_r = min_radius;
        uint32_t local_max_r = max_radius;
        if (num_circles >= 4) {
            local_min_r = (std::max)(2u, grid_size / 16);               // 64 -> 4
            local_max_r = (std::max)(local_min_r + 1, grid_size / (num_circles + 2)); // 64,5 -> 9
        }
        local_max_r = (std::min)(local_max_r, grid_size / 2 - 1);
        
        // Generate random circles (ensuring they fit in grid)
        constexpr bool ENFORCE_SEPARATION = true;
        constexpr uint32_t MAX_TRIES_PER_CIRCLE = 512;
        constexpr uint32_t SEP_MARGIN = 2; // extra separation beyond r_i + r_j
        for (uint32_t c = 0; c < num_circles; ++c) {
            bool placed = false;
            for (uint32_t tries = 0; tries < MAX_TRIES_PER_CIRCLE; ++tries) {
                uint32_t r = local_min_r + (rand() % (local_max_r - local_min_r + 1));
                uint32_t x = r + (rand() % (grid_size - 2 * r));
                uint32_t y = r + (rand() % (grid_size - 2 * r));

                bool ok = true;
                if (ENFORCE_SEPARATION) {
                    for (uint32_t j = 0; j < c; ++j) {
                        int dx = int(x) - int(circles_out[j].x);
                        int dy = int(y) - int(circles_out[j].y);
                        int rr = int(r) + int(circles_out[j].r) + int(SEP_MARGIN);
                        if (dx * dx + dy * dy < rr * rr) {
                            ok = false;
                            break;
                        }
                    }
                }

                if (ok) {
                    circles_out[c].x = uint16_t(x);
                    circles_out[c].y = uint16_t(y);
                    circles_out[c].r = uint16_t(r);
                    placed = true;
                    break;
                }
            }

            // Fallback (should be rare). Place without separation constraints.
            if (!placed) {
                uint32_t r = local_min_r + (rand() % (local_max_r - local_min_r + 1));
                uint32_t x = r + (rand() % (grid_size - 2 * r));
                uint32_t y = r + (rand() % (grid_size - 2 * r));
                circles_out[c].x = uint16_t(x);
                circles_out[c].y = uint16_t(y);
                circles_out[c].r = uint16_t(r);
            }
        }
        
        // Sort circles into a canonical, easy-to-learn order.
        // Lexicographic ordering (x, then y, then r) avoids near-ties from (x+y)
        // and matches the point sorting bias (points are sorted by x/y below).
        std::sort(circles_out, circles_out + num_circles, [](const CircleData& a, const CircleData& b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.r < b.r;
        });
        
        // Pad unused circles with zeros (special "no circle" token)
        for (uint32_t c = num_circles; c < n_max_prims; ++c) {
            circles_out[c].x = 0;
            circles_out[c].y = 0;
            circles_out[c].r = 0;
        }
        
        // Sample points from the circles (with some noise)
        uint32_t points_per_circle = n_points / num_circles;
        uint32_t extra_points = n_points % num_circles;
        
        uint32_t point_idx = 0;
        constexpr float TWO_PI = 6.28318531f;
        
        for (uint32_t c = 0; c < num_circles && point_idx < n_points; ++c) {
            uint32_t pts_for_this = points_per_circle + (c < extra_points ? 1 : 0);
            float cx = float(circles_out[c].x);
            float cy = float(circles_out[c].y);
            float cr = float(circles_out[c].r);
            
            float angle_step = TWO_PI / float(pts_for_this);
            float base_angle = angle_step * (float(rand()) / float(RAND_MAX));

            // NOTE: With many circles, per-circle points become sparse (e.g. ~12 points/circle at 5 circles).
            // Even small noise + quantization can make exact integer circle recovery very hard.
            // For debugging/learning-capacity experiments, we support a near-noiseless mode.
            constexpr bool NOISELESS_POINTS = true;
            if (NOISELESS_POINTS) {
                // Remove random per-circle rotation so the mapping becomes more deterministic.
                // This significantly reduces the effective label noise introduced by quantization.
                base_angle = 0.0f;
            }
            
            for (uint32_t p = 0; p < pts_for_this && point_idx < n_points; ++p) {
                float angle = base_angle + angle_step * float(p);
                if (!NOISELESS_POINTS) {
                    // Keep some randomness, but avoid large jitter/bias that makes the exact
                    // generating circle hard to recover from quantized points.
                    angle += (float(rand()) / float(RAND_MAX) - 0.5f) * (0.05f * angle_step);
                }

                // Radial noise centered around 1.0 (no systematic shrink).
                float noise = 1.0f;
                if (!NOISELESS_POINTS) {
                    noise = 0.99f + 0.02f * float(rand()) / float(RAND_MAX);
                }
                float px = cx + cr * noise * cosf(angle);
                float py = cy + cr * noise * sinf(angle);
                
                // Clamp to grid
                px = fmaxf(0.0f, fminf(float(grid_size - 1), px));
                py = fmaxf(0.0f, fminf(float(grid_size - 1), py));
                
                points_out[point_idx].x = uint16_t(px + 0.5f);
                points_out[point_idx].y = uint16_t(py + 0.5f);
                ++point_idx;
            }
        }
        
        std::sort(points_out, points_out + point_idx, [](const PointData& a, const PointData& b) {
            if (a.x == b.x) return a.y < b.y;
            return a.x < b.x;
        });
        
        // Pad remaining slots (should rarely trigger)
        for (; point_idx < n_points; ++point_idx) {
            points_out[point_idx].x = 0;
            points_out[point_idx].y = 0;
        }

        return num_circles;
    }
    
    // Save visualization to BMP
    void save_sample_bmp(const char* filename, const CircleData* circles, uint32_t num_circles,
                         const PointData* points, uint32_t num_points,
                         const CircleData* predicted = nullptr, uint32_t num_predicted = 0) {
        const int IMG_SIZE = 512;
        const float SCALE = float(IMG_SIZE) / float(grid_size);
        
        BMP bmp(IMG_SIZE, IMG_SIZE);
        bmp.clear(32, 32, 32);  // Dark gray background
        
        // Draw grid
        bmp.draw_grid(int(SCALE * 8), 48, 48, 48);
        
        
        // Draw predicted circles (green outline, if provided)
        for (uint32_t c = 0; c < num_predicted; ++c) {
            if (predicted[c].r == 0) continue;
            int cx = int(predicted[c].x * SCALE);
            int cy = int(predicted[c].y * SCALE);
            int cr = int(predicted[c].r * SCALE);
            bmp.draw_circle(cx, IMG_SIZE - 1 - cy, cr, 0, 200, 0, 2);
        }

        // Draw ground truth circles (white thin outline)
        for (uint32_t c = 0; c < num_circles; ++c) {
            if (circles[c].r == 0) continue;  // Skip "no circle" tokens
            int cx = int(circles[c].x * SCALE);
            int cy = int(circles[c].y * SCALE);
            int cr = int(circles[c].r * SCALE);
            bmp.draw_circle(cx, IMG_SIZE - 1 - cy, cr, 150, 150, 150, 1);
        }
        
        // Draw input points (white)
        for (uint32_t p = 0; p < num_points; ++p) {
            int px = int(points[p].x * SCALE);
            int py = int(points[p].y * SCALE);
            bmp.draw_point(px, IMG_SIZE - 1 - py, 3, 255, 255, 255);
        }
        
        bmp.save(filename);
    }
    
    // Save loss curve to BMP
    static void save_loss_graph(const char* filename, const std::vector<float>& losses, 
                                int width = 800, int height = 400) {
        BMP bmp(width, height);
        bmp.clear(32, 32, 32);  // Dark gray background
        
        if (losses.empty()) return;
        
        // Find min/max loss for scaling
        float min_loss = losses[0], max_loss = losses[0];
        for (float l : losses) {
            min_loss = fminf(min_loss, l);
            max_loss = fmaxf(max_loss, l);
        }
        
        // Add some padding to range
        float range = max_loss - min_loss;
        if (range < 0.001f) range = 1.0f;
        min_loss -= range * 0.1f;
        max_loss += range * 0.1f;
        range = max_loss - min_loss;
        
        const int MARGIN = 40;
        int plot_width = width - 2 * MARGIN;
        int plot_height = height - 2 * MARGIN;
        
        // Draw axes
        for (int x = MARGIN; x < width - MARGIN; ++x) {
            bmp.set_pixel(x, height - MARGIN, 100, 100, 100);
        }
        for (int y = MARGIN; y < height - MARGIN; ++y) {
            bmp.set_pixel(MARGIN, y, 100, 100, 100);
        }
        
        // Draw horizontal grid lines and labels
        for (int i = 0; i <= 4; ++i) {
            int y = MARGIN + i * plot_height / 4;
            for (int x = MARGIN; x < width - MARGIN; x += 4) {
                bmp.set_pixel(x, y, 60, 60, 60);
            }
        }
        
        // Draw raw loss curve (cyan color, dimmer)
        int prev_x = -1, prev_y = -1;
        for (size_t i = 0; i < losses.size(); ++i) {
            int x = MARGIN + int(float(i) / float(losses.size() - 1) * float(plot_width));
            float norm_loss = (losses[i] - min_loss) / range;
            int y = height - MARGIN - int(norm_loss * float(plot_height));
            
            // Clamp y
            y = (std::max)(MARGIN, (std::min)(height - MARGIN, y));
            
            if (prev_x >= 0) {
                bmp.draw_line(prev_x, prev_y, x, y, 0, 100, 150);  // Dimmer cyan for raw
            }
            prev_x = x;
            prev_y = y;
        }
        
        // Draw smoothed loss curve overlay (bright yellow/orange)
        // Using exponential moving average (EMA) for smoothing
        float ema_alpha = 0.05f;  // Smoothing factor (smaller = smoother)
        float ema = losses[0];
        prev_x = -1;
        prev_y = -1;
        for (size_t i = 0; i < losses.size(); ++i) {
            ema = ema_alpha * losses[i] + (1.0f - ema_alpha) * ema;
            
            int x = MARGIN + int(float(i) / float(losses.size() - 1) * float(plot_width));
            float norm_loss = (ema - min_loss) / range;
            int y = height - MARGIN - int(norm_loss * float(plot_height));
            
            // Clamp y
            y = (std::max)(MARGIN, (std::min)(height - MARGIN, y));
            
            if (prev_x >= 0) {
                bmp.draw_line(prev_x, prev_y, x, y, 255, 180, 0);  // Bright orange for smoothed
            }
            prev_x = x;
            prev_y = y;
        }
        
        bmp.save(filename);
        // printf("  Saved loss graph: %s (min=%.4f, max=%.4f)\n", filename, min_loss + range * 0.1f, max_loss - range * 0.1f);
    }
};

// ============================================================================
// Flow Matching Circle Detection (continuous, 10-step sampler)
// ============================================================================

static float rand_uniform_01() {
    return float(rand()) / float(RAND_MAX);
}

// Gaussian random number using Box-Muller transform
static float randn_box_muller() {
    float u1 = (std::max)(1e-7f, rand_uniform_01());
    float u2 = rand_uniform_01();
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530718f * u2;
    return r * cosf(theta);
}

// ============================================================================
// Single-Step Drifting Circle Detection (continuous, conditional)
// ============================================================================

struct DriftMLP {
    uint32_t input_dim = 0;
    uint32_t x_dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t batch_size = 0;

    Graph model;
    Tensor* input = nullptr;   // (B, input_dim)
    Tensor* target = nullptr;  // (B, x_dim)
    Tensor* output = nullptr;  // (B, x_dim)
    Tensor* loss = nullptr;    // (1)

    Tensor* W1 = nullptr;      // (input_dim, hidden_dim)
    Tensor* W2 = nullptr;      // (hidden_dim, x_dim)
    Tensor* Wskip = nullptr;   // (input_dim, x_dim)

    DriftMLP(uint32_t input_dim_, uint32_t x_dim_, uint32_t hidden_dim_, uint32_t batch_size_)
        : input_dim(input_dim_), x_dim(x_dim_), hidden_dim(hidden_dim_), batch_size(batch_size_) {
        build_graph();
    }

    void build_graph() {
        input = &model.tensor({batch_size, input_dim});
        target = &model.tensor({batch_size, x_dim});

        W1 = &model.tensor({input_dim, hidden_dim}, true);
        W2 = &model.tensor({hidden_dim, x_dim}, true);
        Wskip = &model.tensor({input_dim, x_dim}, true);

        Tensor& h = model.matmul(*input, *W1);
        Tensor& h_relu = model.relu(h);
        Tensor& out_main = model.matmul(h_relu, *W2);
        Tensor& out_skip = model.matmul(*input, *Wskip);
        output = &model.add(out_main, out_skip);
        loss = &model.mse_loss(*output, *target);
    }

    void init_weights(uint32_t seed = 42) {
        srand(seed);
        float w1_scale = 0.6f * sqrtf(2.0f / float(input_dim));
        float w2_scale = 0.6f * sqrtf(2.0f / float(hidden_dim));
        float wskip_scale = 0.02f;
        W1->random_init(w1_scale);
        W2->random_init(w2_scale);
        Wskip->random_init(wskip_scale);
    }

    float train_batch(const float16_t* input_in, const float16_t* target_in, float learning_rate) {
        memcpy(input->cpu(), input_in, batch_size * input_dim * sizeof(float16_t));
        memcpy(target->cpu(), target_in, batch_size * x_dim * sizeof(float16_t));
        input->cpu_upload(false);
        target->cpu_upload(false);

        model.eval(true, false, false);
        loss->cpu_download(false);

        constexpr float BETA1 = 0.9f;
        constexpr float BETA2 = 0.999f;
        constexpr float EPS   = 1e-4f;
        for (auto* param : model.params) {
            assert(param->grad_tensor);
            evk::ai::adam(*param, param->grad(), model.adam_states[param], learning_rate, BETA1, BETA2, EPS);
        }
        evk::ai::SubmitCmd(true);

        return float(loss->cpu()[0]);
    }

    void forward_batch(const float16_t* input_in, float16_t* out) {
        memcpy(input->cpu(), input_in, batch_size * input_dim * sizeof(float16_t));
        input->cpu_upload(false);
        model.eval(false);
        output->cpu_download(true);
        memcpy(out, output->cpu(), batch_size * x_dim * sizeof(float16_t));
    }
};

struct CircleDriftModel {
    uint32_t n_points = 0;
    uint32_t n_max_prims = 0;
    uint32_t grid_size = 0;

    static constexpr uint32_t COND_FEATS_PER_CLUSTER = 4; // (cx, cy, r_est, count)
    static constexpr uint32_t RAW_POINT_FEATS = 2;        // (x, y)
    static constexpr float PRESENCE_SCALE = 2.0f;
    uint32_t cond_dim_raw = 0;
    uint32_t x_dim_raw = 0;
    uint32_t cond_dim = 0;   // padded to 16 for matmul
    uint32_t x_dim = 0;      // padded to 16 for matmul
    uint32_t noise_dim_raw = 0;
    uint32_t noise_dim = 0;  // padded to 16
    uint32_t input_dim = 0;

    DriftMLP net;

    std::vector<CircleData> tmp_circles;
    std::vector<PointData> tmp_points;

    std::vector<float16_t> batch_cond;
    std::vector<float> batch_cond_raw;
    std::vector<float> batch_x1_raw;
    std::vector<float> batch_x_base_raw;
    std::vector<float16_t> batch_input;
    std::vector<float16_t> batch_output;
    std::vector<float> batch_x_pred_raw;
    std::vector<float> batch_drift_raw;
    std::vector<float16_t> batch_target;

    CircleDriftModel(uint32_t n_points_, uint32_t n_max_prims_, uint32_t grid_size_, uint32_t hidden_dim_, uint32_t batch_size_)
        : n_points(n_points_), n_max_prims(n_max_prims_), grid_size(grid_size_),
          cond_dim_raw(n_max_prims_ * COND_FEATS_PER_CLUSTER + n_points_ * RAW_POINT_FEATS),
          x_dim_raw(4 * n_max_prims_),
          cond_dim(((cond_dim_raw + 15u) / 16u) * 16u),
          x_dim(((x_dim_raw + 15u) / 16u) * 16u),
          noise_dim_raw(4 * n_max_prims_),
          noise_dim(((noise_dim_raw + 15u) / 16u) * 16u),
          input_dim(cond_dim + noise_dim),
          net(input_dim, x_dim, hidden_dim_, batch_size_) {
        tmp_circles.resize(n_max_prims);
        tmp_points.resize(n_points);
        batch_cond.resize(net.batch_size * cond_dim);
        batch_cond_raw.resize(net.batch_size * cond_dim_raw);
        batch_x1_raw.resize(net.batch_size * x_dim_raw);
        batch_x_base_raw.resize(net.batch_size * x_dim_raw);
        batch_input.resize(net.batch_size * input_dim);
        batch_output.resize(net.batch_size * x_dim);
        batch_x_pred_raw.resize(net.batch_size * x_dim_raw);
        batch_drift_raw.resize(net.batch_size * x_dim_raw);
        batch_target.resize(net.batch_size * x_dim);
    }

    void init_weights(uint32_t seed = 42) { net.init_weights(seed); }

    float norm_xy(uint16_t v) const {
        float f = float(v) / float(grid_size - 1);
        return f * 2.0f - 1.0f;
    }

    float norm_xy_f(float v) const {
        float f = v / float(grid_size - 1);
        return f * 2.0f - 1.0f;
    }

    float norm_r(uint16_t r) const {
        uint32_t r_max_u = (std::max)(1u, grid_size / 2 - 1);
        float f = float(r) / float(r_max_u);
        f = fmaxf(0.0f, fminf(1.0f, f));
        return f * 2.0f - 1.0f;
    }

    float norm_r_f(float r) const {
        uint32_t r_max_u = (std::max)(1u, grid_size / 2 - 1);
        float f = r / float(r_max_u);
        f = fmaxf(0.0f, fminf(1.0f, f));
        return f * 2.0f - 1.0f;
    }

    uint16_t denorm_xy(float x) const {
        float f = (x + 1.0f) * 0.5f * float(grid_size - 1);
        int iv = int(floorf(f + 0.5f));
        iv = (std::max)(0, (std::min)(iv, int(grid_size - 1)));
        return uint16_t(iv);
    }

    uint16_t denorm_r(float x) const {
        uint32_t r_max_u = (std::max)(1u, grid_size / 2 - 1);
        float f = (x + 1.0f) * 0.5f * float(r_max_u);
        int iv = int(floorf(f + 0.5f));
        iv = (std::max)(0, (std::min)(iv, int(r_max_u)));
        return uint16_t(iv);
    }

    void encode_cond(float16_t* out, const PointData* pts) {
        for (uint32_t i = 0; i < cond_dim; ++i) out[i] = float16_t(0.0f);

        const uint32_t K = n_max_prims;
        std::vector<float> cx(K), cy(K);
        std::vector<float> sumx(K), sumy(K);
        std::vector<uint32_t> cnt(K);
        std::vector<uint32_t> assign(n_points);

        std::vector<uint32_t> chosen;
        chosen.reserve(K);
        chosen.push_back(0u);
        cx[0] = float(pts[0].x);
        cy[0] = float(pts[0].y);
        for (uint32_t k = 1; k < K; ++k) {
            float best_min_d2 = -1.0f;
            uint32_t best_p = 0u;
            for (uint32_t p = 0; p < n_points; ++p) {
                float px = float(pts[p].x);
                float py = float(pts[p].y);
                float min_d2 = 1e30f;
                for (uint32_t j = 0; j < chosen.size(); ++j) {
                    float dx = px - cx[j];
                    float dy = py - cy[j];
                    float d2 = dx * dx + dy * dy;
                    min_d2 = (std::min)(min_d2, d2);
                }
                if (min_d2 > best_min_d2) {
                    best_min_d2 = min_d2;
                    best_p = p;
                }
            }
            chosen.push_back(best_p);
            cx[k] = float(pts[best_p].x);
            cy[k] = float(pts[best_p].y);
        }

        constexpr uint32_t ITERS = 16;
        for (uint32_t it = 0; it < ITERS; ++it) {
            std::fill(sumx.begin(), sumx.end(), 0.0f);
            std::fill(sumy.begin(), sumy.end(), 0.0f);
            std::fill(cnt.begin(), cnt.end(), 0u);

            for (uint32_t p = 0; p < n_points; ++p) {
                float px = float(pts[p].x);
                float py = float(pts[p].y);
                float best_d2 = 1e30f;
                uint32_t best_k = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    float dx = px - cx[k];
                    float dy = py - cy[k];
                    float d2 = dx * dx + dy * dy;
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_k = k;
                    }
                }
                assign[p] = best_k;
                sumx[best_k] += px;
                sumy[best_k] += py;
                cnt[best_k]++;
            }

            for (uint32_t k = 0; k < K; ++k) {
                if (cnt[k] > 0) {
                    cx[k] = sumx[k] / float(cnt[k]);
                    cy[k] = sumy[k] / float(cnt[k]);
                }
            }
        }

        std::vector<float> rmean(K, 0.0f);
        for (uint32_t p = 0; p < n_points; ++p) {
            uint32_t k = assign[p];
            float px = float(pts[p].x);
            float py = float(pts[p].y);
            float dx = px - cx[k];
            float dy = py - cy[k];
            rmean[k] += sqrtf(dx * dx + dy * dy);
        }
        for (uint32_t k = 0; k < K; ++k) {
            if (cnt[k] > 0) rmean[k] /= float(cnt[k]);
        }

        std::vector<float> r = rmean;
        constexpr uint32_t EM_ITERS = 12;
        for (uint32_t it = 0; it < EM_ITERS; ++it) {
            std::fill(sumx.begin(), sumx.end(), 0.0f);
            std::fill(sumy.begin(), sumy.end(), 0.0f);
            std::fill(cnt.begin(), cnt.end(), 0u);

            for (uint32_t p = 0; p < n_points; ++p) {
                float px = float(pts[p].x);
                float py = float(pts[p].y);
                float best_cost = 1e30f;
                uint32_t best_k = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    float dx = px - cx[k];
                    float dy = py - cy[k];
                    float d = sqrtf(dx * dx + dy * dy);
                    float cost = fabsf(d - r[k]);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_k = k;
                    }
                }
                assign[p] = best_k;
                sumx[best_k] += px;
                sumy[best_k] += py;
                cnt[best_k]++;
            }

            for (uint32_t k = 0; k < K; ++k) {
                if (cnt[k] > 0) {
                    cx[k] = sumx[k] / float(cnt[k]);
                    cy[k] = sumy[k] / float(cnt[k]);
                } else {
                    uint32_t idx = (k * (n_points - 1)) / (K ? K : 1u);
                    cx[k] = float(pts[idx].x);
                    cy[k] = float(pts[idx].y);
                }
            }

            std::fill(rmean.begin(), rmean.end(), 0.0f);
            for (uint32_t p = 0; p < n_points; ++p) {
                uint32_t k = assign[p];
                float px = float(pts[p].x);
                float py = float(pts[p].y);
                float dx = px - cx[k];
                float dy = py - cy[k];
                rmean[k] += sqrtf(dx * dx + dy * dy);
            }
            for (uint32_t k = 0; k < K; ++k) {
                if (cnt[k] > 0) rmean[k] /= float(cnt[k]);
            }

            r = rmean;
        }

        struct ClusterFeat { float x, y, r; uint32_t count; };
        std::vector<ClusterFeat> feats(K);
        for (uint32_t k = 0; k < K; ++k) feats[k] = {cx[k], cy[k], rmean[k], cnt[k]};
        std::sort(feats.begin(), feats.end(), [](const ClusterFeat& a, const ClusterFeat& b) {
            if (a.count == 0 && b.count > 0) return false;
            if (a.count > 0 && b.count == 0) return true;
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });

        for (uint32_t k = 0; k < K; ++k) {
            float fx = (feats[k].count > 0) ? norm_xy_f(feats[k].x) : 0.0f;
            float fy = (feats[k].count > 0) ? norm_xy_f(feats[k].y) : 0.0f;
            float fr = (feats[k].count > 0) ? norm_r_f(feats[k].r) : -1.0f;
            float fc = (float(feats[k].count) / float(n_points)) * 2.0f - 1.0f;

            out[k * COND_FEATS_PER_CLUSTER + 0] = float16_t(fx);
            out[k * COND_FEATS_PER_CLUSTER + 1] = float16_t(fy);
            out[k * COND_FEATS_PER_CLUSTER + 2] = float16_t(fr);
            out[k * COND_FEATS_PER_CLUSTER + 3] = float16_t(fc);
        }

        // Append raw (x, y) point list as additional conditioning features.
        uint32_t base = n_max_prims * COND_FEATS_PER_CLUSTER;
        for (uint32_t p = 0; p < n_points; ++p) {
            out[base + p * RAW_POINT_FEATS + 0] = float16_t(norm_xy(pts[p].x));
            out[base + p * RAW_POINT_FEATS + 1] = float16_t(norm_xy(pts[p].y));
        }
    }

    void encode_x1(float* out, const CircleData* circles) {
        for (uint32_t i = 0; i < x_dim_raw; ++i) out[i] = 0.0f;
        for (uint32_t c = 0; c < n_max_prims; ++c) {
            out[c * 4 + 0] = norm_xy(circles[c].x);
            out[c * 4 + 1] = norm_xy(circles[c].y);
            out[c * 4 + 2] = norm_r(circles[c].r);
            out[c * 4 + 3] = (circles[c].r > 0) ? PRESENCE_SCALE : -PRESENCE_SCALE;
        }
    }

    void encode_x_base_from_cond(float* out, const float16_t* cond_feat) const {
        for (uint32_t i = 0; i < x_dim_raw; ++i) out[i] = 0.0f;
        uint32_t C = (std::min)(n_max_prims, x_dim_raw / 4u);
        for (uint32_t c = 0; c < C; ++c) {
            out[c * 4 + 0] = float(cond_feat[c * 4 + 0]);
            out[c * 4 + 1] = float(cond_feat[c * 4 + 1]);
            out[c * 4 + 2] = float(cond_feat[c * 4 + 2]);
            out[c * 4 + 3] = float(cond_feat[c * 4 + 3]) * PRESENCE_SCALE;
        }
    }

    void decode_x1_to_circles(const float* x, CircleData* circles_out) const {
        constexpr float PRESENCE_THRESHOLD = 0.5f * PRESENCE_SCALE;
        for (uint32_t c = 0; c < n_max_prims; ++c) {
            float r_norm = x[c * 4 + 2];
            float pres = x[c * 4 + 3];
            if (pres < PRESENCE_THRESHOLD || r_norm < -0.75f) {
                circles_out[c].x = 0;
                circles_out[c].y = 0;
                circles_out[c].r = 0;
                continue;
            }

            uint16_t xx = denorm_xy(x[c * 4 + 0]);
            uint16_t yy = denorm_xy(x[c * 4 + 1]);
            uint16_t rr = denorm_r(r_norm);
            if (rr == 0) rr = 1;

            circles_out[c].x = xx;
            circles_out[c].y = yy;
            circles_out[c].r = rr;
        }
        std::sort(circles_out, circles_out + n_max_prims, [](const CircleData& a, const CircleData& b) {
            bool a_blank = (a.r == 0);
            bool b_blank = (b.r == 0);
            if (a_blank != b_blank) return b_blank;
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.r < b.r;
        });
    }

    float score_points_to_circles(const PointData* points, const CircleData* circles) const {
        float total = 0.0f;
        for (uint32_t p = 0; p < n_points; ++p) {
            float px = float(points[p].x);
            float py = float(points[p].y);
            float best = 1e30f;
            for (uint32_t c = 0; c < n_max_prims; ++c) {
                if (circles[c].r == 0) continue;
                float cx = float(circles[c].x);
                float cy = float(circles[c].y);
                float cr = float(circles[c].r);
                float dx = px - cx;
                float dy = py - cy;
                float d = sqrtf(dx * dx + dy * dy);
                float cost = fabsf(d - cr);
                best = fminf(best, cost);
            }
            total += best;
        }
        return total;
    }

    void filter_circles_by_support(const PointData* points, CircleData* circles, uint32_t min_support, float tol) const {
        for (uint32_t c = 0; c < n_max_prims; ++c) {
            if (circles[c].r == 0) continue;
            uint32_t support = 0;
            float cx = float(circles[c].x);
            float cy = float(circles[c].y);
            float cr = float(circles[c].r);
            for (uint32_t p = 0; p < n_points; ++p) {
                float px = float(points[p].x);
                float py = float(points[p].y);
                float dx = px - cx;
                float dy = py - cy;
                float d = sqrtf(dx * dx + dy * dy);
                if (fabsf(d - cr) <= tol) support++;
            }
            if (support < min_support) {
                circles[c].x = 0;
                circles[c].y = 0;
                circles[c].r = 0;
            }
        }

        std::sort(circles, circles + n_max_prims, [](const CircleData& a, const CircleData& b) {
            bool a_blank = (a.r == 0);
            bool b_blank = (b.r == 0);
            if (a_blank != b_blank) return b_blank;
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.r < b.r;
        });
    }

    void refine_circles_to_points(const PointData* points, CircleData* circles, uint32_t iters = 10) const {
        uint32_t r_max_u = (std::max)(1u, grid_size / 2 - 1);

        auto clamp_circle = [&](CircleData& c) {
            if (c.r == 0) c.r = 1;
            if (c.r > r_max_u) c.r = uint16_t(r_max_u);
            uint16_t r = c.r;
            if (c.x < r) c.x = r;
            if (c.y < r) c.y = r;
            uint16_t hi = uint16_t(grid_size - 1);
            if (c.x > uint16_t(hi - r)) c.x = uint16_t(hi - r);
            if (c.y > uint16_t(hi - r)) c.y = uint16_t(hi - r);
        };

        for (uint32_t c = 0; c < n_max_prims; ++c) {
            if (circles[c].r == 0) circles[c].r = 1;
            clamp_circle(circles[c]);
        }

        for (uint32_t it = 0; it < iters; ++it) {
            for (uint32_t ci = 0; ci < n_max_prims; ++ci) {
                CircleData best_circle = circles[ci];
                float best_score = score_points_to_circles(points, circles);

                CircleData cand = best_circle;
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dr = -1; dr <= 1; ++dr) {
                            if (dx == 0 && dy == 0 && dr == 0) continue;
                            cand = best_circle;
                            {
                                int ix = int(cand.x) + dx;
                                int iy = int(cand.y) + dy;
                                ix = (std::max)(0, (std::min)(ix, int(grid_size - 1)));
                                iy = (std::max)(0, (std::min)(iy, int(grid_size - 1)));
                                cand.x = uint16_t(ix);
                                cand.y = uint16_t(iy);
                                cand.r = uint16_t((std::max)(1, int(cand.r) + dr));
                            }
                            clamp_circle(cand);

                            CircleData saved = circles[ci];
                            circles[ci] = cand;
                            float sc = score_points_to_circles(points, circles);
                            circles[ci] = saved;

                            if (sc < best_score) {
                                best_score = sc;
                                best_circle = cand;
                            }
                        }
                    }
                }
                circles[ci] = best_circle;
            }
        }

        std::sort(circles, circles + n_max_prims, [](const CircleData& a, const CircleData& b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.r < b.r;
        });
    }

    void compute_drift_block(uint32_t B,
                             const float* x_pred,
                             const float* cond_raw,
                             const float* y_pos,
                             const std::vector<float>& taus,
                             float* drift_out) const {
        std::fill(drift_out, drift_out + B * x_dim_raw, 0.0f);

        const uint32_t P = B;
        const uint32_t N = B;
        const uint32_t M = P + N;
        const uint32_t feat_dim = cond_dim_raw + x_dim_raw;
        const bool mask_self = (B > 1);

        std::vector<float> logits(B * M);
        std::vector<float> row_prob(B * M);
        std::vector<float> col_prob(B * M);

        for (float tau : taus) {
            float tau_eff = tau * sqrtf(float(feat_dim));
            if (tau_eff < 1e-6f) tau_eff = 1e-6f;

            for (uint32_t i = 0; i < B; ++i) {
                const float* ci = cond_raw + i * cond_dim_raw;
                const float* xi = x_pred + i * x_dim_raw;

                for (uint32_t j = 0; j < P; ++j) {
                    const float* cj = cond_raw + j * cond_dim_raw;
                    const float* yj = y_pos + j * x_dim_raw;
                    float dist2 = 0.0f;
                    for (uint32_t d = 0; d < cond_dim_raw; ++d) {
                        float diff = ci[d] - cj[d];
                        dist2 += diff * diff;
                    }
                    for (uint32_t d = 0; d < x_dim_raw; ++d) {
                        float diff = xi[d] - yj[d];
                        dist2 += diff * diff;
                    }
                    float dist = sqrtf(dist2 + 1e-8f);
                    logits[i * M + j] = -dist / tau_eff;
                }

                for (uint32_t k = 0; k < N; ++k) {
                    const float* ck = cond_raw + k * cond_dim_raw;
                    const float* xk = x_pred + k * x_dim_raw;
                    float dist2 = 0.0f;
                    for (uint32_t d = 0; d < cond_dim_raw; ++d) {
                        float diff = ci[d] - ck[d];
                        dist2 += diff * diff;
                    }
                    for (uint32_t d = 0; d < x_dim_raw; ++d) {
                        float diff = xi[d] - xk[d];
                        dist2 += diff * diff;
                    }
                    float dist = sqrtf(dist2 + 1e-8f);
                    float logit = -dist / tau_eff;
                    if (mask_self && k == i) logit = -1e9f;
                    logits[i * M + P + k] = logit;
                }
            }

            for (uint32_t i = 0; i < B; ++i) {
                float max_logit = -1e30f;
                for (uint32_t m = 0; m < M; ++m) {
                    max_logit = (std::max)(max_logit, logits[i * M + m]);
                }
                float sum = 0.0f;
                for (uint32_t m = 0; m < M; ++m) {
                    float e = expf(logits[i * M + m] - max_logit);
                    row_prob[i * M + m] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / (sum + 1e-8f);
                for (uint32_t m = 0; m < M; ++m) {
                    row_prob[i * M + m] *= inv_sum;
                }
            }

            for (uint32_t m = 0; m < M; ++m) {
                float max_logit = -1e30f;
                for (uint32_t i = 0; i < B; ++i) {
                    max_logit = (std::max)(max_logit, logits[i * M + m]);
                }
                float sum = 0.0f;
                for (uint32_t i = 0; i < B; ++i) {
                    float e = expf(logits[i * M + m] - max_logit);
                    col_prob[i * M + m] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / (sum + 1e-8f);
                for (uint32_t i = 0; i < B; ++i) {
                    col_prob[i * M + m] *= inv_sum;
                }
            }

            for (uint32_t i = 0; i < B; ++i) {
                float sum_pos = 0.0f;
                float sum_neg = 0.0f;
                for (uint32_t j = 0; j < P; ++j) {
                    float w = sqrtf(row_prob[i * M + j] * col_prob[i * M + j]);
                    sum_pos += w;
                }
                for (uint32_t k = 0; k < N; ++k) {
                    float w = sqrtf(row_prob[i * M + P + k] * col_prob[i * M + P + k]);
                    sum_neg += w;
                }
                float inv_pos = 1.0f / (sum_pos + 1e-8f);
                float inv_neg = 1.0f / (sum_neg + 1e-8f);

                for (uint32_t d = 0; d < x_dim_raw; ++d) {
                    float acc_pos = 0.0f;
                    float acc_neg = 0.0f;
                    for (uint32_t j = 0; j < P; ++j) {
                        float w = sqrtf(row_prob[i * M + j] * col_prob[i * M + j]) * inv_pos;
                        acc_pos += w * y_pos[j * x_dim_raw + d];
                    }
                    for (uint32_t k = 0; k < N; ++k) {
                        float w = sqrtf(row_prob[i * M + P + k] * col_prob[i * M + P + k]) * inv_neg;
                        acc_neg += w * x_pred[k * x_dim_raw + d];
                    }
                    drift_out[i * x_dim_raw + d] += (acc_pos - acc_neg);
                }
            }
        }
    }

    float train_step(CircleDataset& dataset, float lr, uint32_t force_num_circles,
                     const std::vector<float>& taus, float noise_scale, uint32_t group_size, float drift_blend) {
        if (group_size == 0) group_size = 1;
        uint32_t groups = (net.batch_size + group_size - 1) / group_size;

        std::vector<float16_t> cond_tmp(cond_dim, float16_t(0.0f));
        std::vector<float> x1_tmp(x_dim_raw, 0.0f);
        std::vector<float> xbase_tmp(x_dim_raw, 0.0f);
        std::vector<float> cond_raw_tmp(cond_dim_raw, 0.0f);

        for (uint32_t g = 0; g < groups; ++g) {
            dataset.generate_sample(tmp_circles.data(), tmp_points.data(), force_num_circles);

            std::fill(cond_tmp.begin(), cond_tmp.end(), float16_t(0.0f));
            encode_cond(cond_tmp.data(), tmp_points.data());

            std::fill(x1_tmp.begin(), x1_tmp.end(), 0.0f);
            encode_x1(x1_tmp.data(), tmp_circles.data());

            std::fill(xbase_tmp.begin(), xbase_tmp.end(), 0.0f);
            encode_x_base_from_cond(xbase_tmp.data(), cond_tmp.data());

            for (uint32_t i = 0; i < cond_dim_raw; ++i) {
                cond_raw_tmp[i] = float(cond_tmp[i]);
            }

            for (uint32_t s = 0; s < group_size; ++s) {
                uint32_t b = g * group_size + s;
                if (b >= net.batch_size) break;

                memcpy(batch_cond.data() + b * cond_dim, cond_tmp.data(), cond_dim * sizeof(float16_t));
                memcpy(batch_x1_raw.data() + b * x_dim_raw, x1_tmp.data(), x_dim_raw * sizeof(float));
                memcpy(batch_x_base_raw.data() + b * x_dim_raw, xbase_tmp.data(), x_dim_raw * sizeof(float));
                memcpy(batch_cond_raw.data() + b * cond_dim_raw, cond_raw_tmp.data(), cond_dim_raw * sizeof(float));

                float16_t* inp = batch_input.data() + b * input_dim;
                memcpy(inp, cond_tmp.data(), cond_dim * sizeof(float16_t));
                for (uint32_t i = 0; i < noise_dim; ++i) {
                    float n = (i < noise_dim_raw) ? (randn_box_muller() * noise_scale) : 0.0f;
                    inp[cond_dim + i] = float16_t(n);
                }
            }
        }

        net.forward_batch(batch_input.data(), batch_output.data());
        for (uint32_t b = 0; b < net.batch_size; ++b) {
            for (uint32_t d = 0; d < x_dim_raw; ++d) {
                float delta = float(batch_output[b * x_dim + d]);
                batch_x_pred_raw[b * x_dim_raw + d] = batch_x_base_raw[b * x_dim_raw + d] + delta;
            }
        }

        std::fill(batch_drift_raw.begin(), batch_drift_raw.end(), 0.0f);
        uint32_t groups_for_drift = (net.batch_size + group_size - 1) / group_size;
        for (uint32_t g = 0; g < groups_for_drift; ++g) {
            uint32_t b0 = g * group_size;
            uint32_t count = (std::min)(group_size, net.batch_size - b0);
            compute_drift_block(count,
                                batch_x_pred_raw.data() + b0 * x_dim_raw,
                                batch_cond_raw.data() + b0 * cond_dim_raw,
                                batch_x1_raw.data() + b0 * x_dim_raw,
                                taus,
                                batch_drift_raw.data() + b0 * x_dim_raw);
        }

        constexpr float DRIFT_STEP = 1.0f;
        constexpr float DRIFT_RMS_CLIP = 1.25f;
        for (uint32_t b = 0; b < net.batch_size; ++b) {
            float rms = 0.0f;
            for (uint32_t d = 0; d < x_dim_raw; ++d) {
                float v = batch_drift_raw[b * x_dim_raw + d];
                rms += v * v;
            }
            rms = sqrtf(rms / float((std::max)(1u, x_dim_raw)));
            float scale = (rms > DRIFT_RMS_CLIP) ? (DRIFT_RMS_CLIP / rms) : 1.0f;

            for (uint32_t d = 0; d < x_dim_raw; ++d) {
                float drifted = batch_x_pred_raw[b * x_dim_raw + d] + DRIFT_STEP * scale * batch_drift_raw[b * x_dim_raw + d];
                float clamp = ((d % 4u) == 3u) ? (1.2f * PRESENCE_SCALE) : 1.2f;
                drifted = fmaxf(-clamp, fminf(clamp, drifted));
                float gt = batch_x1_raw[b * x_dim_raw + d];
                float x_target = drift_blend * drifted + (1.0f - drift_blend) * gt;
                float delta_target = x_target - batch_x_base_raw[b * x_dim_raw + d];
                batch_target[b * x_dim + d] = float16_t(delta_target);
            }
            for (uint32_t d = x_dim_raw; d < x_dim; ++d) {
                batch_target[b * x_dim + d] = batch_output[b * x_dim + d];
            }
        }

        return net.train_batch(batch_input.data(), batch_target.data(), lr);
    }

    void predict(const PointData* points, CircleData* circles_out, uint32_t samples, float noise_scale) {
        std::vector<float16_t> cond(cond_dim);
        encode_cond(cond.data(), points);

        std::vector<float> x_base(x_dim_raw, 0.0f);
        encode_x_base_from_cond(x_base.data(), cond.data());

        std::vector<float16_t> input(net.batch_size * input_dim, float16_t(0.0f));
        std::vector<float16_t> output(net.batch_size * x_dim);
        std::vector<float> x_raw(x_dim_raw);
        std::vector<CircleData> cand(n_max_prims);

        constexpr float SUPPORT_TOL = 1.5f;
        constexpr uint32_t MIN_SUPPORT = 4;

        float best_score = 1e30f;
        std::vector<CircleData> best(n_max_prims);

        // Baseline candidate from conditioning only.
        decode_x1_to_circles(x_base.data(), cand.data());
        refine_circles_to_points(points, cand.data(), 35);
        filter_circles_by_support(points, cand.data(), MIN_SUPPORT, SUPPORT_TOL);
        best_score = score_points_to_circles(points, cand.data());
        best = cand;

        for (uint32_t s = 0; s < samples; ++s) {
            float16_t* inp = input.data();
            memcpy(inp, cond.data(), cond_dim * sizeof(float16_t));
            for (uint32_t i = 0; i < noise_dim; ++i) {
                float n = (i < noise_dim_raw) ? (randn_box_muller() * noise_scale) : 0.0f;
                inp[cond_dim + i] = float16_t(n);
            }

            net.forward_batch(input.data(), output.data());
            for (uint32_t d = 0; d < x_dim_raw; ++d) {
                float delta = float(output[d]);
                x_raw[d] = x_base[d] + delta;
            }

            decode_x1_to_circles(x_raw.data(), cand.data());
            refine_circles_to_points(points, cand.data(), 35);
            filter_circles_by_support(points, cand.data(), MIN_SUPPORT, SUPPORT_TOL);
            float sc = score_points_to_circles(points, cand.data());
            if (sc < best_score) {
                best_score = sc;
                best = cand;
            }
        }

        memcpy(circles_out, best.data(), n_max_prims * sizeof(CircleData));
    }
};

void run_circle_drifting() {
    printf("\n=== Circle Detection: Single-Step Drifting ===\n");

    constexpr uint32_t N_POINTS = 64;
    constexpr uint32_t N_MAX_PRIMS = 6;
    constexpr uint32_t GRID_SIZE = 64;
    constexpr uint32_t HIDDEN = 512;
    constexpr uint32_t BATCH = 128;

    const uint32_t MIN_CIRCLES = 3;
    const uint32_t MAX_CIRCLES = 6;

    const int MAX_STEPS = 12000;
    const int LOG_INTERVAL = 200;
    const int EVAL_INTERVAL = 400;
    const int WARMUP_STEPS = 0;
    const float MAX_DRIFT_BLEND = 0.0f;

    const float LR = 0.0010f;
    const float NOISE_SCALE = 0.0f;
    const std::vector<float> TAUS = {0.2f, 0.4f, 0.8f};
    const uint32_t NUM_SAMPLES = 1;
    const uint32_t GROUP_SIZE = 4;

    const float TARGET_AVG_ERR = 7.0f;
    const float TARGET_EXTRA = 0.05f;
    const float TARGET_EXACT = 0.94f;

        printf("  Config: n_points=%u n_max_prims=%u circles=[%u..%u] grid=%u hidden=%u batch=%u\n",
            N_POINTS, N_MAX_PRIMS, MIN_CIRCLES, MAX_CIRCLES, GRID_SIZE, HIDDEN, BATCH);
        printf("  Training: steps=%d lr=%.6f noise=%.3f group=%u warmup=%d max_blend=%.2f\n",
            MAX_STEPS, LR, NOISE_SCALE, GROUP_SIZE, WARMUP_STEPS, MAX_DRIFT_BLEND);

    const unsigned SEED_WEIGHTS = 42;
    const unsigned SEED_DATA = 1337;
    const unsigned SEED_EVAL = 12345;

    CircleDataset dataset(N_POINTS, N_MAX_PRIMS, GRID_SIZE);
    CircleDriftModel model(N_POINTS, N_MAX_PRIMS, GRID_SIZE, HIDDEN, BATCH);
    model.init_weights(SEED_WEIGHTS);
    srand(SEED_DATA);

    struct EvalStats {
        double avg_abs_err = 0.0;
        double avg_extra = 0.0;
        double exact_frac = 0.0;
    };

    auto sample_num_circles = [&]() -> uint32_t {
        uint32_t lo = (std::min)(MIN_CIRCLES, N_MAX_PRIMS);
        uint32_t hi = (std::min)((std::max)(MIN_CIRCLES, MAX_CIRCLES), N_MAX_PRIMS);
        if (lo > hi) (std::swap)(lo, hi);
        uint32_t span = hi - lo + 1;
        return lo + (span > 0 ? (rand() % span) : 0u);
    };

    auto evaluate = [&](int num_test) -> EvalStats {
        std::vector<CircleData> gt(N_MAX_PRIMS);
        std::vector<CircleData> pred(N_MAX_PRIMS);
        std::vector<PointData> pts(N_POINTS);

        long long total_abs_err = 0;
        long long total_extra = 0;
        int total_exact = 0;

        for (int t = 0; t < num_test; ++t) {
            uint32_t k = dataset.generate_sample(gt.data(), pts.data(), sample_num_circles());
            model.predict(pts.data(), pred.data(), NUM_SAMPLES, NOISE_SCALE);

            bool exact = true;
            int abs_err = 0;
            int extra = 0;
            for (uint32_t c = 0; c < k; ++c) {
                int err = abs(int(gt[c].x) - int(pred[c].x)) +
                          abs(int(gt[c].y) - int(pred[c].y)) +
                          abs(int(gt[c].r) - int(pred[c].r));
                abs_err += err;
                if (err != 0) exact = false;
            }
            for (uint32_t c = k; c < N_MAX_PRIMS; ++c) {
                if (pred[c].r > 0) {
                    exact = false;
                    extra++;
                }
            }
            total_abs_err += abs_err;
            total_extra += extra;
            if (exact) total_exact++;
        }

        EvalStats st;
        st.avg_abs_err = double(total_abs_err) / double(num_test);
        st.avg_extra = double(total_extra) / double(num_test);
        st.exact_frac = double(total_exact) / double(num_test);
        return st;
    };

    std::vector<float> loss_hist;
    loss_hist.reserve(MAX_STEPS);

    std::vector<std::vector<float16_t>> best_weights;
    std::vector<uint32_t> best_counts;
    bool have_best = false;
    double best_err = 1e9;
    double best_exact = 0.0;

    auto save_weights = [&]() {
        best_weights.clear();
        best_counts.clear();
        for (auto* param : model.net.model.params) {
            uint32_t count = param->shape.count();
            std::vector<float16_t> buf(count);
            param->cpu_download(true);
            memcpy(buf.data(), param->cpu(), count * sizeof(float16_t));
            best_weights.push_back(std::move(buf));
            best_counts.push_back(count);
        }
        have_best = true;
    };

    auto restore_weights = [&]() {
        if (!have_best) return;
        for (size_t i = 0; i < model.net.model.params.size(); ++i) {
            auto* param = model.net.model.params[i];
            uint32_t count = best_counts[i];
            memcpy(param->cpu(), best_weights[i].data(), count * sizeof(float16_t));
            param->cpu_upload(false);
        }
        evk::ai::SubmitCmd(true);
    };

    for (int step = 0; step < MAX_STEPS; ++step) {
        float lr = LR;
        if (step < 200) lr = LR * float(step + 1) / 200.0f;

        float drift_blend = (WARMUP_STEPS > 0) ? (float(step) / float(WARMUP_STEPS)) : 1.0f;
        if (drift_blend > 1.0f) drift_blend = 1.0f;
        if (drift_blend > MAX_DRIFT_BLEND) drift_blend = MAX_DRIFT_BLEND;
        uint32_t train_circles = sample_num_circles();
        float loss = model.train_step(dataset, lr, train_circles, TAUS, NOISE_SCALE, GROUP_SIZE, drift_blend);
        loss_hist.push_back(loss);

        if (step % LOG_INTERVAL == 0 || step == MAX_STEPS - 1) {
            printf("  step %4d: drift_loss=%.6f\n", step, loss);
            CircleDataset::save_loss_graph("drift_loss_graph.bmp", loss_hist);
        }

        if (step % EVAL_INTERVAL == 0 || step == MAX_STEPS - 1) {
                 srand(SEED_EVAL + step);
                EvalStats st = evaluate(200);
                 srand(SEED_DATA + step + 1);
            printf("  eval: avg_abs_err=%.4f avg_extra=%.4f exact=%.2f%%\n",
                   st.avg_abs_err, st.avg_extra, 100.0 * st.exact_frac);

                if (st.avg_abs_err < best_err || (fabs(st.avg_abs_err - best_err) < 1e-6 && st.exact_frac > best_exact)) {
                    best_err = st.avg_abs_err;
                    best_exact = st.exact_frac;
                    save_weights();
                }

            if (st.avg_abs_err <= TARGET_AVG_ERR && st.avg_extra <= TARGET_EXTRA && st.exact_frac >= TARGET_EXACT) {
                printf("  target reached; stopping early.\n");
                break;
            }
        }
    }

    restore_weights();
    srand(SEED_EVAL);
    EvalStats final_stats = evaluate(400);
    printf("\n  === Final Evaluation ===\n");
    printf("  Avg abs err (GT circles only): %.4f\n", final_stats.avg_abs_err);
    printf("  Avg extra circles: %.4f\n", final_stats.avg_extra);
    printf("  Exact samples: %.2f%%\n", 100.0 * final_stats.exact_frac);

    const int NUM_VIS = 6;
    std::vector<CircleData> vis_gt(N_MAX_PRIMS);
    std::vector<CircleData> vis_pred(N_MAX_PRIMS);
    std::vector<PointData> vis_pts(N_POINTS);
    for (int i = 0; i < NUM_VIS; ++i) {
        uint32_t k = dataset.generate_sample(vis_gt.data(), vis_pts.data(), sample_num_circles());
        model.predict(vis_pts.data(), vis_pred.data(), NUM_SAMPLES, NOISE_SCALE);
        char filename[64];
        snprintf(filename, sizeof(filename), "drift_result_%02d.bmp", i);
        dataset.save_sample_bmp(filename, vis_gt.data(), k, vis_pts.data(), N_POINTS, vis_pred.data(), N_MAX_PRIMS);
    }
    printf("  Saved %d result images: drift_result_00.bmp .. drift_result_%02d.bmp\n", NUM_VIS, NUM_VIS - 1);
}
