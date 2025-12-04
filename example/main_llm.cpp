#include <iostream>
#include <evk_ai.h>

#include "bmp.h"

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

// ============================================================================
// Transformer Model
// ============================================================================

// Attention block: Query-Only attention + FFN with residuals
// Uses Query-Only Attention (https://arxiv.org/pdf/2510.00365)
struct AttentionBlock {
    // Weight tensors (owned by graph)
    Tensor* w_q = nullptr;  // Query projection
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
        w1 = &graph.tensor({embed_dim, hidden_dim}, true);
        w2 = &graph.tensor({hidden_dim, embed_dim}, true);
    }
    
    // Initialize weights with random values using proper scaling for deep networks
    // Uses a combination of He initialization with residual scaling
    void init_weights(float base_scale, bool use_residual_scaling = true) {
        // He initialization: scale by sqrt(2/fan_in)
        float w_q_scale = base_scale * sqrtf(2.0f / float(embed_dim));
        float w1_scale = base_scale * sqrtf(2.0f / float(embed_dim));
        
        // For the last projection in residual path, scale down to prevent growth
        // This is the "residual scaling" technique from GPT-2 / muP
        float residual_scale = use_residual_scaling ? (1.0f / sqrtf(float(2 * total_layers))) : 1.0f;
        float w2_scale = base_scale * sqrtf(2.0f / float(hidden_dim)) * residual_scale;
        
        w_q->random_init(w_q_scale);
        w1->random_init(w1_scale);
        w2->random_init(w2_scale);
    }
    
    // Forward pass through this block
    // input: (B, N, embed_dim)
    // Returns: (B, N, embed_dim) output with residual connections
    Tensor& forward(Graph& graph, Tensor& input, float residual_scale = 1.0f) {
        // Pre-norm Transformer block:
        //   y = x + scale * Attn(RMSNorm(x))
        //   z = y + scale * FFN(RMSNorm(y))

        // 1) Pre-norm attention block
        Tensor& norm1 = graph.rms_norm(input);
        
        // Attention projections (Query-Only: K and V use the same input)
        Tensor& q = graph.matmul(norm1, *w_q);
        Tensor& k = norm1;
        Tensor& v = norm1;
        
        // Causal self-attention with residual
        Tensor& attn_out = graph.causal_attention(q, k, v);
        
        // Optional: scale the attention output before residual (helps with deep networks)
        Tensor* attn_scaled = &attn_out;
        if (residual_scale != 1.0f) {
            attn_scaled = &graph.scale(attn_out, residual_scale);
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
        
        // Compute residual scale for deep networks (prevents gradient explosion/vanishing)
        // This scales down each residual contribution so total variance stays bounded
        float residual_scale = 1.0f;
        if (num_layers > 2) {
            // Use alpha = 1/sqrt(2*num_layers) as suggested by various papers on deep transformers
            residual_scale = 1.0f / sqrtf(float(num_layers));
        }
        
        // Pass through N attention blocks
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &blocks[i].forward(model, *x, residual_scale);
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
        float residual_scale = 1.0f;
        if (num_layers > 2) {
            residual_scale = 1.0f / sqrtf(float(num_layers));
        }
        
        // Forward pass (same architecture as training)
        Tensor& embedded = inference.embed(*inf_token_emb, *inf_input);
        Tensor& input_with_pos = inference.add_position_embedding(embedded, *inf_pos_emb, 1, seq_len);
        
        // Pass through N attention blocks (use same residual_scale as training)
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &inf_blocks[i].forward(inference, *x, residual_scale);
        }

        inf_logits = &inference.matmul(*x, *inf_w_out);
    }
    
    void init_weights(uint32_t seed = 42) {
        srand(seed);
        
        // Proper initialization for deep transformers:
        // - Embeddings: use small values to keep activations bounded
        // - Weights: use He-like scaling with residual compensation
        float emb_scale = 0.02f;  // Small embedding init
        float base_scale = 1.0f;  // Base scale for weights (blocks will apply He scaling)
        
        token_emb->random_init(emb_scale);
        pos_emb->random_init(emb_scale);
        
        // Initialize all attention block weights with proper scaling
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init_weights(base_scale, true);  // Use residual scaling
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
        
        input_tokens->cpu_upload();
        targets->cpu_upload();
        
        model.eval(true);
        
        loss->cpu_download();
        float loss_val = float(loss->cpu()[0]);

        model.step_adam(learning_rate, 0.9f, 0.999f, 1e-4f);
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

// Helper class for text-based transformer usage with vocabulary
struct TextTransformer {
    Transformer transformer;
    const char* vocab;
    uint32_t actual_vocab_size;
    
    TextTransformer(uint32_t vocab_size, uint32_t seq_len, uint32_t embed_dim, 
                    uint32_t hidden_dim, uint32_t batch_size, uint32_t num_layers = 1,
                    const char* vocab = nullptr)
        : transformer(vocab_size, seq_len, embed_dim, hidden_dim, batch_size, num_layers),
          vocab(vocab)
    {
        actual_vocab_size = vocab ? (uint32_t)strlen(vocab) : vocab_size;
    }
    
    void init_weights(uint32_t seed = 42) { transformer.init_weights(seed); }
    
    uint16_t char_to_token(char c) const {
        if (!vocab) return uint16_t(c);
        for (uint32_t i = 0; vocab[i]; ++i) {
            if (vocab[i] == c) return uint16_t(i);
        }
        return 0;
    }
    
    char token_to_char(uint16_t t) const {
        if (!vocab) return char(t);
        return (t < actual_vocab_size) ? vocab[t] : '?';
    }
    
    // Train on a single batch of string samples
    float train_batch(const char** samples, uint32_t batch_start, float learning_rate) {
        uint32_t seq_len = transformer.seq_len;
        uint32_t batch_size = transformer.batch_size;
        
        std::vector<uint16_t> inp(batch_size * seq_len);
        std::vector<uint16_t> tgt(batch_size * seq_len);
        
        for (uint32_t b = 0; b < batch_size; ++b) {
            const char* sample = samples[batch_start + b];
            uint32_t len = (uint32_t)strlen(sample);
            
            for (uint32_t t = 0; t < seq_len; ++t) {
                inp[b * seq_len + t] = (t < len) ? char_to_token(sample[t]) : uint16_t(0);
                tgt[b * seq_len + t] = (t + 1 < len) ? char_to_token(sample[t + 1]) : uint16_t(0);
            }
        }
        
        return transformer.train_batch(inp.data(), tgt.data(), learning_rate);
    }
    
    // Train for multiple epochs over all string samples
    float train(const char** samples, uint32_t num_samples, int epochs, 
                float learning_rate = 0.01f, int log_interval = 10) {
        float last_loss = 0.0f;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int num_batches = 0;
            
            for (uint32_t batch_start = 0; batch_start + transformer.batch_size <= num_samples; batch_start += transformer.batch_size) {
                epoch_loss += train_batch(samples, batch_start, learning_rate);
                num_batches++;
            }
            
            last_loss = epoch_loss / float(num_batches);
            
            if (log_interval > 0 && (epoch % log_interval == 0 || epoch == epochs - 1)) {
                printf("  epoch %3d: loss = %.4f\n", epoch, last_loss);
            }
        }
        
        return last_loss;
    }
    
    // Generate text autoregressively from a prefix string
    std::string generate(const char* prefix, int max_new_tokens) {
        uint32_t prefix_len = (uint32_t)strlen(prefix);
        std::vector<uint16_t> prefix_tokens(prefix_len);
        for (uint32_t i = 0; i < prefix_len; ++i) {
            prefix_tokens[i] = char_to_token(prefix[i]);
        }
        
        std::vector<uint16_t> output(transformer.seq_len);
        uint32_t total_len = transformer.generate(prefix_tokens.data(), prefix_len, output.data(), max_new_tokens);
        
        std::string result;
        for (uint32_t i = 0; i < total_len; ++i) {
            result += token_to_char(output[i]);
        }
        return result;
    }
};


const char* samples[] = {
    "lorem ipsum dolor sit amet.  ",
    "consectetur adipiscing elit. ",
    "sed do eiusmod tempor incid. ",
    "incididunt ut labore dolore. ",
    "dolore magna aliqua ut enim. ",
    "enim ad minim veniam quis.   ",
    "quis nostrud exercitation.   ",
    "ullamco laboris nisi aliquip.",
    "ex ea commodo consequat.     ",
    "duis aute irure dolor in.    ",
    "in reprehenderit in voluptate",
    "velit esse cillum dolore eu. ",
    "eu fugiat nulla pariatur.    ",
    "excepteur sint occaecat cupid",
    "cupidatat non proident sunt. ",
    "sunt in culpa qui officia.   ",
    "officia deserunt mollit anim.",
    "anim id est laborum lorem.   ",
    "ipsum dolor sit amet consec. ",
    "adipiscing elit sed do eius. ",
    "eiusmod tempor incididunt ut.",
    "labore et dolore magna aliq. ",
    "ut enim ad minim veniam qu.  ",
    "quis nostrud exercitation ul.",
    "lamco laboris nisi aliquip ex",
    "ea commodo consequat duis a. ",
    "ute irure dolor in repreh.   ",
    "in voluptate velit esse ci.  ",
    "illum dolore eu fugiat nu.   ",
    "lla pariatur excepteur sin.  ",
    "t occaecat cupidatat non p.  ",
    "roident sunt in culpa qui o. ",
    "fficia deserunt mollit ani.  ",
    "nim id est laborum lorem.    ",
    "ipsum dolor sit amet co.     ",
    "nsectetur adipiscing elits.  ",
    "ed do eiusmod tempor in.     ",
    "cididunt ut labore et do.    ",
    "lore magna aliqua ut en.     ",
    "im ad minim veniam quis n.   ",
    "ostrud exercitation ullam.   ",
    "co laboris nisi aliquip e.   ",
    "x ea commodo consequat d.    ",
    "uis aute irure dolor i.      ",
    "n reprehenderit in volup.    ",
    "tate velit esse cillum d.    ",
    "olore eu fugiat nulla p.     ",
    "ariatur excepteur sint o.    ",
    "ccaecat cupidatat non p.     ",
    "roident sunt in culpa q.     ",
};
const uint32_t NUM_SAMPLES = sizeof(samples) / sizeof(samples[0]);

// Returns final loss for comparison
float run_next_token_prediction_attention() {
    printf("\n=== Attention-based Transformer ===\n");
    
    const char* VOCAB = " abcdefghijklmnopqrstuvwxyz.,\n";
    const uint32_t VOCAB_SIZE = 32;
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    const uint32_t NUM_LAYERS = 1;  // Number of attention blocks
    
    printf("  Attention Model: vocab=%u, seq=%u, embed=%u, hidden=%u, layers=%u\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
    
    // Create and initialize transformer
    TextTransformer transformer(VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, NUM_LAYERS, VOCAB);
    transformer.init_weights(42);
    
    const int EPOCHS = 60;
    const float LR = 0.01f;
    
    printf("  Training for %d epochs...\n", EPOCHS);
    
    // Train the model
    float last_loss = transformer.train(samples, NUM_SAMPLES, EPOCHS, LR, 10);
    
    // === Autoregressive Generation ===
    printf("\n  === Autoregressive Generation ===\n");
    
    const char* prefixes[] = {
        "lorem ",
        "dolor ",
        "in ",
        "qu",
        "ex ",
    };
    const int NUM_PREFIXES = sizeof(prefixes) / sizeof(prefixes[0]);
    const int GEN_LEN = 28;
    
    for (int p = 0; p < NUM_PREFIXES; ++p) {
        std::string generated = transformer.generate(prefixes[p], GEN_LEN);
        printf("  \"%s\" -> \"%s\"\n", prefixes[p], generated.c_str());
    }
    
    return last_loss;
}

// Forward declaration
void run_circle_detection(uint32_t num_layers);

#include <chrono>
void main_llm() {
    evk::ai::initialize();
    printf("=== main_llm ===\n");
    // run_next_token_prediction_attention();  // Comment out for now

    auto start = std::chrono::high_resolution_clock::now();
    // run_circle_detection(1);
    // run_circle_detection(2);
    // run_circle_detection(4);
    run_circle_detection(8);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("run_circle_detection() took %.4f seconds\n", duration.count());
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
    uint32_t generate_sample(CircleData* circles_out, PointData* points_out) {
#if 0 // Enable variable number of circles for more variety
        uint32_t num_circles = 1 + (rand() % n_max_prims);
#else // Fixed number of circles
        uint32_t num_circles = n_max_prims;
#endif
        
        // Generate random circles (ensuring they fit in grid)
        for (uint32_t c = 0; c < num_circles; ++c) {
            uint32_t r = min_radius + (rand() % (max_radius - min_radius + 1));
            uint32_t x = r + (rand() % (grid_size - 2 * r));
            uint32_t y = r + (rand() % (grid_size - 2 * r));
            circles_out[c].x = uint16_t(x);
            circles_out[c].y = uint16_t(y);
            circles_out[c].r = uint16_t(r);
        }
        
        // Sort circles by (x + y) for simple consistent ordering
        // This gives a clean diagonal sweep ordering that's easier to learn
        for (uint32_t i = 0; i < num_circles; ++i) {
            for (uint32_t j = i + 1; j < num_circles; ++j) {
                uint32_t sum_i = circles_out[i].x + circles_out[i].y;
                uint32_t sum_j = circles_out[j].x + circles_out[j].y;
                if (sum_j < sum_i) {
                    CircleData tmp = circles_out[i];
                    circles_out[i] = circles_out[j];
                    circles_out[j] = tmp;
                }
            }
        }
        
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
        for (uint32_t c = 0; c < num_circles && point_idx < n_points; ++c) {
            uint32_t pts_for_this = points_per_circle + (c < extra_points ? 1 : 0);
            float cx = float(circles_out[c].x);
            float cy = float(circles_out[c].y);
            float cr = float(circles_out[c].r);
            
            for (uint32_t p = 0; p < pts_for_this && point_idx < n_points; ++p) {
                // Sample angle uniformly
                float angle = 2.0f * 3.14159265f * float(rand()) / float(RAND_MAX);
                // Add small noise to radius
                float noise = 0.95f + 0.01f * float(rand()) / float(RAND_MAX);
                float px = cx + cr * noise * cosf(angle);
                float py = cy + cr * noise * sinf(angle);
                
                // Clamp to grid
                px = fmaxf(0.0f, fminf(float(grid_size - 1), px));
                py = fmaxf(0.0f, fminf(float(grid_size - 1), py));
                
                points_out[point_idx].x = uint16_t(px + 0.5f);
                points_out[point_idx].y = uint16_t(py + 0.5f);
                point_idx++;
            }
        }

        // TODO: sort the points by x axis
        for (uint32_t i = 0; i < n_points; ++i) {
            for (uint32_t j = i + 1; j < n_points; ++j) {
                if (points_out[i].x > points_out[j].x) {
                    PointData temp = points_out[i];
                    points_out[i] = points_out[j];
                    points_out[j] = temp;
                }
            }
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

// Helper functions for computing CircleDetector dimensions
inline uint32_t compute_vocab_size(uint32_t grid_size) {
    uint32_t raw_vocab = grid_size + 1;  // +1 for "no value" token (0)
    return ((raw_vocab + 15) / 16) * 16;  // Pad to multiple of 16
}

inline uint32_t compute_total_seq_len(uint32_t n_points, uint32_t n_max_prims) {
    uint32_t input_seq_len = n_points * 2;  // x, y for each point
    uint32_t output_seq_len = n_max_prims * 3;  // x, y, r for each circle
    uint32_t raw_total = input_seq_len + output_seq_len;
    return ((raw_total + 15) / 16) * 16;  // Pad to multiple of 16
}

// Circle detection using Transformer
// Input: N_POINTS 2D points -> encoded as tokens
// Output: N_MAX_PRIMS * 3 values (x, y, r for each circle)
struct CircleDetector {
    // Hyperparameters
    uint32_t n_points;          // Number of input points
    uint32_t n_max_prims;       // Max number of output circles
    uint32_t grid_size;         // Discrete coordinate space
    
    // Computed dimensions
    uint32_t vocab_size;        // grid_size for x, y, r
    uint32_t input_seq_len;     // n_points * 2 (x, y pairs)
    uint32_t output_seq_len;    // n_max_prims * 3 (x, y, r per circle)
    uint32_t total_seq_len;     // input + output sequence
    
    // Underlying transformer
    Transformer transformer;
    
    CircleDetector(uint32_t n_points_, uint32_t n_max_prims_, uint32_t grid_size_,
                   uint32_t embed_dim_, uint32_t hidden_dim_, uint32_t batch_size_,
                   uint32_t num_layers_ = 2)
        : n_points(n_points_), n_max_prims(n_max_prims_), grid_size(grid_size_),
          vocab_size(compute_vocab_size(grid_size_)),
          input_seq_len(n_points_ * 2),
          output_seq_len(n_max_prims_ * 3),
          total_seq_len(compute_total_seq_len(n_points_, n_max_prims_)),
          transformer(compute_vocab_size(grid_size_), compute_total_seq_len(n_points_, n_max_prims_), 
                     embed_dim_, hidden_dim_, batch_size_, num_layers_)
    {
    }
    
    void init_weights(uint32_t seed = 42) {
        transformer.init_weights(seed);
    }
    
    // Encode points into input tokens
    void encode_points(uint16_t* tokens, const PointData* points) {
        for (uint32_t p = 0; p < n_points; ++p) {
            // +1 because 0 is reserved for "no value"
            tokens[p * 2 + 0] = points[p].x + 1;
            tokens[p * 2 + 1] = points[p].y + 1;
        }
    }
    
    // Encode circles into target tokens
    void encode_circles(uint16_t* tokens, const CircleData* circles, uint32_t num_circles) {
        for (uint32_t c = 0; c < n_max_prims; ++c) {
            if (c < num_circles && circles[c].r > 0) {
                tokens[c * 3 + 0] = circles[c].x + 1;
                tokens[c * 3 + 1] = circles[c].y + 1;
                tokens[c * 3 + 2] = circles[c].r + 1;
            } else {
                // "No circle" token
                tokens[c * 3 + 0] = 0;
                tokens[c * 3 + 1] = 0;
                tokens[c * 3 + 2] = 0;
            }
        }
    }
    
    // Decode tokens back to circles
    void decode_circles(const uint16_t* tokens, CircleData* circles_out) {
        for (uint32_t c = 0; c < n_max_prims; ++c) {
            uint16_t x = tokens[c * 3 + 0];
            uint16_t y = tokens[c * 3 + 1];
            uint16_t r = tokens[c * 3 + 2];
            circles_out[c].x = (x > 0) ? uint16_t(x - 1) : 0;
            circles_out[c].y = (y > 0) ? uint16_t(y - 1) : 0;
            circles_out[c].r = (r > 0) ? uint16_t(r - 1) : 0;
        }
    }
    
    // Train on a batch of generated samples
    float train_batch(CircleDataset& dataset, float learning_rate) {
        std::vector<CircleData> circles(n_max_prims);
        std::vector<PointData> points(n_points);
        
        std::vector<uint16_t> full_inp(transformer.batch_size * total_seq_len);
        std::vector<uint16_t> tgt(transformer.batch_size * total_seq_len);
        
        for (uint32_t b = 0; b < transformer.batch_size; ++b) {
            uint32_t num_circles = dataset.generate_sample(circles.data(), points.data());
            
            // Encode input points
            std::vector<uint16_t> point_tokens(input_seq_len);
            encode_points(point_tokens.data(), points.data());
            
            // Encode target circles
            std::vector<uint16_t> circle_tokens(output_seq_len);
            encode_circles(circle_tokens.data(), circles.data(), num_circles);
            
            // Build full sequence: [point_tokens, shifted_circle_tokens]
            // For teacher forcing: input[i] predicts target[i]
            // Input:  [p0, p1, ..., pN-1, c0, c1, ..., cM-2, cM-1]
            // Target: [p1, p2, ..., pN,   c0, c1, ..., cM-1, EOS]
            // But we only care about predicting circle tokens
            
            // Full input sequence
            for (uint32_t t = 0; t < input_seq_len; ++t) {
                full_inp[b * total_seq_len + t] = point_tokens[t];
            }
            // Shifted target for output positions (input to transformer at output positions)
            // Position input_seq_len + i gets token at output position i-1 (or start token)
            full_inp[b * total_seq_len + input_seq_len] = 0;  // Start token
            for (uint32_t t = 1; t < output_seq_len; ++t) {
                full_inp[b * total_seq_len + input_seq_len + t] = circle_tokens[t - 1];
            }
            
            // Target: shift by 1 (predict next token)
            // For input positions, use target=0 (IGNORE token) to exclude from loss
            // For output positions, target is the circle token sequence
            for (uint32_t t = 0; t < input_seq_len; ++t) {
                tgt[b * total_seq_len + t] = 0;  // IGNORE token - no loss computed here
            }
            for (uint32_t t = 0; t < output_seq_len; ++t) {
                tgt[b * total_seq_len + input_seq_len + t] = circle_tokens[t];
            }
            // Pad remaining positions with zeros
            for (uint32_t t = input_seq_len + output_seq_len; t < total_seq_len; ++t) {
                full_inp[b * total_seq_len + t] = 0;
                tgt[b * total_seq_len + t] = 0;
            }
        }
        
        return transformer.train_batch(full_inp.data(), tgt.data(), learning_rate);
    }
    
    // Predict circles from points (autoregressive generation)
    void predict(const PointData* points, CircleData* circles_out) {
        transformer.copy_weights_to_inference();
        
        uint16_t* inp = (uint16_t*)transformer.inf_input->cpu();
        
        // Encode input points
        std::vector<uint16_t> point_tokens(input_seq_len);
        encode_points(point_tokens.data(), points);
        
        // Fill input sequence
        for (uint32_t t = 0; t < input_seq_len; ++t) {
            inp[t] = point_tokens[t];
        }
        
        // Autoregressive generation for output tokens
        inp[input_seq_len] = 0;  // Start token
        for (uint32_t t = input_seq_len + 1; t < total_seq_len; ++t) {
            inp[t] = 0;
        }
        
        std::vector<uint16_t> generated_tokens(output_seq_len);
        
        for (uint32_t out_pos = 0; out_pos < output_seq_len; ++out_pos) {
            transformer.inf_input->cpu_upload();
            transformer.inference.eval(false);
            
            transformer.inf_logits->cpu_download();
            float16_t* logits = transformer.inf_logits->cpu();
            
            // Get prediction at current output position
            uint32_t pos = input_seq_len + out_pos;
            float max_val = -1e9f;
            uint16_t max_idx = 0;
            for (uint32_t v = 0; v < vocab_size; ++v) {
                float val = float(logits[pos * vocab_size + v]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = uint16_t(v);
                }
            }
            
            generated_tokens[out_pos] = max_idx;
            
            // Feed back for next position
            if (out_pos + 1 < output_seq_len) {
                inp[input_seq_len + out_pos + 1] = max_idx;
            }
        }
        
        decode_circles(generated_tokens.data(), circles_out);
    }
};

void debug_gradients(Transformer& transformer, const char* label) {
    printf("  [%s] Gradient Statistics:\n", label);
    
    // Token embedding gradients
    if (transformer.token_emb->grad_tensor) {
        transformer.token_emb->grad().stats().print("token_emb.grad");
    }
    
    // Positional embedding gradients
    if (transformer.pos_emb->grad_tensor) {
        transformer.pos_emb->grad().stats().print("pos_emb.grad");
    }
    
    // Output projection gradients
    if (transformer.w_out->grad_tensor) {
        transformer.w_out->grad().stats().print("w_out.grad");
    }
    
    // First block's weights
    for( size_t i = 0; i < transformer.blocks.size(); ++i) {
        auto& block = transformer.blocks[i];
        printf("  -- Block %zu --\n", i);
        if (block.w_q->grad_tensor) {
            block.w_q->grad().stats().print("w_q.grad");
        }
        if (block.w1->grad_tensor) {
            block.w1->grad().stats().print("w1.grad");
        }
        if (block.w2->grad_tensor) {
            block.w2->grad().stats().print("w2.grad");
        }
    }
    
    printf("\n");
}

// Main function for circle detection demo
void run_circle_detection(uint32_t num_layers) {
    printf("\n=== Circle Detection Transformer ===\n");
    
    // Hyperparameters - optimized for 2-circle detection
    constexpr uint32_t N_POINTS = 32;       // Points sampled from circles
    constexpr uint32_t N_MAX_PRIMS = 2;     // Detect up to 1 circle
    constexpr uint32_t GRID_SIZE = 64;      // Discrete coordinate space
    constexpr uint32_t EMBED_DIM = 160;     // Embedding dimension
    constexpr uint32_t HIDDEN_DIM = 320;    // FFN hidden dimension
    constexpr uint32_t BATCH_SIZE = 96;     // Batch size
    uint32_t NUM_LAYERS = 8;                // Number of transformer layers
    
    printf("  Config: n_points=%u, n_max_prims=%u, grid=%u, embed=%u, hidden=%u, layers=%u\n",
           N_POINTS, N_MAX_PRIMS, GRID_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
    
    // Create dataset generator
    CircleDataset dataset(N_POINTS, N_MAX_PRIMS, GRID_SIZE);
    
    // Create model
    CircleDetector detector(N_POINTS, N_MAX_PRIMS, GRID_SIZE, EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, NUM_LAYERS);
    detector.init_weights(42);
    
    printf("  Input seq len: %u, Output seq len: %u, Total: %u\n",
           detector.input_seq_len, detector.output_seq_len, detector.total_seq_len);
    
    // Training hyperparameters
    const int EPOCHS = 1000;
    const float LR = 0.010f;
    const int WARMUP_EPOCHS = 100;
    
    printf("  Training for %d epochs with LR=%.4f, warmup=%d...\n", EPOCHS, LR, WARMUP_EPOCHS);
    
    std::vector<float> loss_history;
    loss_history.reserve(EPOCHS);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Learning rate schedule: linear warmup then cosine decay
        float min_lr = LR * 0.05f;
        float effective_lr = LR;
        if (epoch < WARMUP_EPOCHS) {
            effective_lr = LR * float(epoch + 1) / float(WARMUP_EPOCHS);
        } else {
            float progress = float(epoch - WARMUP_EPOCHS) / float(EPOCHS - WARMUP_EPOCHS);
            effective_lr = min_lr + (LR - min_lr) * 0.5f * (1.0f + cosf(3.14159265f * progress));
        }
        
        float epoch_loss = detector.train_batch(dataset, effective_lr);
        loss_history.push_back(epoch_loss);
        
        if (epoch % 100 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, epoch_loss);
            CircleDataset::save_loss_graph("loss_graph.bmp", loss_history);
        }
        
        // Debug gradients (uncomment to debug gradient issues)
        if (epoch == 0 || (epoch % 1 == 0) || epoch == EPOCHS - 1) {
            // debug_gradients(detector.transformer, ("epoch " + std::to_string(epoch)).c_str());
        }
    }
    
    
    // Evaluation and visualization
    printf("\n  === Evaluation ===\n");
    printf("  error must be zero to match exactly the validation\n");
    
    // Generate test samples and visualize predictions
    const int NUM_TEST = 5;
    std::vector<CircleData> gt_circles(N_MAX_PRIMS);
    std::vector<CircleData> pred_circles(N_MAX_PRIMS);
    std::vector<PointData> points(N_POINTS);
    
    srand(12345);  // Different seed for test samples

    int total_error = 0;
    for (int t = 0; t < NUM_TEST; ++t) {
        uint32_t num_gt = dataset.generate_sample(gt_circles.data(), points.data());
        
        printf("  Test %d (num_circles=%u): \n", t, num_gt);
        detector.predict(points.data(), pred_circles.data());
        
        // Print validation with detailed debug info
        for (uint32_t c = 0; c < num_gt; ++c) {
            int err = abs(gt_circles[c].x - pred_circles[c].x) + 
                      abs(gt_circles[c].y - pred_circles[c].y) + 
                      abs(gt_circles[c].r - pred_circles[c].r);
            printf("    Circle %d: GT=(%d,%d,r=%d) Pred=(%d,%d,r=%d) error=%d\n", 
                   c, gt_circles[c].x, gt_circles[c].y, gt_circles[c].r,
                   pred_circles[c].x, pred_circles[c].y, pred_circles[c].r, err);
            total_error += err;
        }
        // Also show any extra predicted circles
        for (uint32_t c = num_gt; c < N_MAX_PRIMS; ++c) {
            if (pred_circles[c].r > 0) {
                printf("    Circle %d (extra): Pred=(%d,%d,r=%d)\n",
                       c, pred_circles[c].x, pred_circles[c].y, pred_circles[c].r);
            }
        }

        // Save visualization
        char filename[64];
        snprintf(filename, sizeof(filename), "circle_test_%d.bmp", t);
        dataset.save_sample_bmp(filename, gt_circles.data(), num_gt, 
                               points.data(), N_POINTS,
                               pred_circles.data(), N_MAX_PRIMS);
    }
    printf("  Total error: %d\n", total_error);
}
