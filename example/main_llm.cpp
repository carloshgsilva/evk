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
    
    AttentionBlock() : embed_dim(0), hidden_dim(0) {}
    
    // Initialize weights in the given graph
    void init(Graph& graph, uint32_t embed_dim_, uint32_t hidden_dim_) {
        embed_dim = embed_dim_;
        hidden_dim = hidden_dim_;
        
        w_q = &graph.tensor({embed_dim, embed_dim}, true);
        w1 = &graph.tensor({embed_dim, hidden_dim}, true);
        w2 = &graph.tensor({hidden_dim, embed_dim}, true);
    }
    
    // Initialize weights with random values
    void init_weights(float scale) {
        w_q->random_init(scale);
        w1->random_init(scale);
        w2->random_init(scale);
    }
    
    // Forward pass through this block
    // input: (B, N, embed_dim)
    // Returns: (B, N, embed_dim) output with residual connections
    Tensor& forward(Graph& graph, Tensor& input) {
        // Attention projections (Query-Only: K and V are just input)
        Tensor& q = graph.matmul(input, *w_q);
        Tensor& k = input;
        Tensor& v = input;
        
        // Causal self-attention with residual
        Tensor& attn_out = graph.causal_attention(q, k, v);
        Tensor& attn_residual = graph.add(input, attn_out);
        
        // FFN with residual
        Tensor& hidden = graph.matmul(attn_residual, *w1);
        Tensor& hidden_relu = graph.relu(hidden);
        Tensor& hidden_proj = graph.matmul(hidden_relu, *w2);
        Tensor& output = graph.add(attn_residual, hidden_proj);
        
        return output;
    }
};

// Query-Only Attention Transformer with N layers
struct Transformer {
    // Hyperparameters
    uint32_t vocab_size;
    uint32_t seq_len;
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t batch_size;
    uint32_t num_layers;
    
    // Vocabulary
    const char* vocab;
    uint32_t actual_vocab_size;
    
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
                uint32_t hidden_dim, uint32_t batch_size, uint32_t num_layers = 1,
                const char* vocab = nullptr)
        : vocab_size(vocab_size), seq_len(seq_len), embed_dim(embed_dim),
          hidden_dim(hidden_dim), batch_size(batch_size), num_layers(num_layers), vocab(vocab)
    {
        actual_vocab_size = vocab ? (uint32_t)strlen(vocab) : vocab_size;
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
        
        // Initialize attention blocks
        blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init(model, embed_dim, hidden_dim);
        }
        
        // Forward pass
        Tensor& embedded = model.embed(*token_emb, *input_tokens);
        Tensor& input_with_pos = model.add_position_embedding(embedded, *pos_emb, batch_size, seq_len);
        
        // Pass through N attention blocks
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &blocks[i].forward(model, *x);
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
            inf_blocks[i].init(inference, embed_dim, hidden_dim);
        }
        
        // Forward pass (same architecture as training)
        Tensor& embedded = inference.embed(*inf_token_emb, *inf_input);
        Tensor& input_with_pos = inference.add_position_embedding(embedded, *inf_pos_emb, 1, seq_len);
        
        // Pass through N attention blocks
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &inf_blocks[i].forward(inference, *x);
        }
        
        inf_logits = &inference.matmul(*x, *inf_w_out);
    }
    
    void init_weights(uint32_t seed = 42) {
        srand(seed);
        float scale = 0.1f / sqrtf(float(embed_dim));
        token_emb->random_init(0.1f);
        pos_emb->random_init(0.1f);
        
        // Initialize all attention block weights (must be before w_out to match original order)
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init_weights(scale);
        }
        
        w_out->random_init(scale);
    }
    
    void copy_weights_to_inference() {
        evk::CmdCopy(token_emb->buffer, inf_token_emb->buffer, token_emb->shape.count() * sizeof(float16_t));
        evk::CmdCopy(pos_emb->buffer, inf_pos_emb->buffer, pos_emb->shape.count() * sizeof(float16_t));
        evk::CmdCopy(w_out->buffer, inf_w_out->buffer, w_out->shape.count() * sizeof(float16_t));
        
        // Copy all attention block weights
        for (uint32_t i = 0; i < num_layers; ++i) {
            evk::CmdCopy(blocks[i].w_q->buffer, inf_blocks[i].w_q->buffer, blocks[i].w_q->shape.count() * sizeof(float16_t));
            evk::CmdCopy(blocks[i].w1->buffer, inf_blocks[i].w1->buffer, blocks[i].w1->shape.count() * sizeof(float16_t));
            evk::CmdCopy(blocks[i].w2->buffer, inf_blocks[i].w2->buffer, blocks[i].w2->shape.count() * sizeof(float16_t));
        }
        evk::Sync();
    }
    
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
    
    // Train on a single batch of samples
    // samples: array of strings (must have at least batch_size elements starting at batch_start)
    // Returns the loss value for this batch
    float train_batch(const char** samples, uint32_t batch_start, float learning_rate) {
        uint16_t* inp = (uint16_t*)input_tokens->cpu();
        uint16_t* tgt = (uint16_t*)targets->cpu();
        
        for (uint32_t b = 0; b < batch_size; ++b) {
            const char* sample = samples[batch_start + b];
            uint32_t len = (uint32_t)strlen(sample);
            
            for (uint32_t t = 0; t < seq_len; ++t) {
                inp[b * seq_len + t] = (t < len) ? char_to_token(sample[t]) : uint16_t(0);
                tgt[b * seq_len + t] = (t + 1 < len) ? char_to_token(sample[t + 1]) : uint16_t(0);
            }
        }
        input_tokens->cpu_upload();
        targets->cpu_upload();
        
        model.eval(true);
        evk::Sync();
        
        loss->cpu_download();
        float loss_val = float(loss->cpu()[0]);
        
        model.step_adam(-learning_rate);
        evk::Sync();
        
        return loss_val;
    }
    
    // Train for multiple epochs over all samples
    // samples: array of training strings
    // num_samples: total number of samples
    // epochs: number of training epochs
    // learning_rate: Adam learning rate
    // log_interval: print loss every N epochs (0 to disable)
    // Returns final average loss
    float train(const char** samples, uint32_t num_samples, int epochs, 
                float learning_rate = 0.01f, int log_interval = 10) {
        float last_loss = 0.0f;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int num_batches = 0;
            
            for (uint32_t batch_start = 0; batch_start + batch_size <= num_samples; batch_start += batch_size) {
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
    
    // Generate tokens autoregressively from a prefix
    // prefix: starting string
    // max_new_tokens: maximum number of new tokens to generate
    // Returns the generated string (prefix + generated tokens)
    std::string generate(const char* prefix, int max_new_tokens) {
        // Ensure inference weights are up to date
        copy_weights_to_inference();
        
        uint32_t prefix_len = (uint32_t)strlen(prefix);
        
        // Initialize sequence with prefix
        std::string generated(prefix);
        uint32_t cur_len = prefix_len;
        
        for (int g = 0; g < max_new_tokens && cur_len < seq_len; ++g) {
            // Fill input tokens
            uint16_t* inp = (uint16_t*)inf_input->cpu();
            for (uint32_t t = 0; t < seq_len; ++t) {
                inp[t] = (t < cur_len) ? char_to_token(generated[t]) : uint16_t(0);
            }
            inf_input->cpu_upload();
            
            // Forward pass (no backward)
            inference.eval(false);
            evk::Sync();
            
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
            generated += token_to_char(max_idx);
            cur_len++;
        }
        
        return generated;
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
    Transformer transformer(VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, NUM_LAYERS, VOCAB);
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
void run_circle_detection();

void main_llm() {
    printf("=== main_llm ===\n");
    // run_next_token_prediction_attention();  // Comment out for now
    run_circle_detection();
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
        // Random number of circles (1 to n_max_prims)
        uint32_t num_circles = 1 + (rand() % n_max_prims);
        
        // Generate random circles (ensuring they fit in grid)
        for (uint32_t c = 0; c < num_circles; ++c) {
            uint32_t r = min_radius + (rand() % (max_radius - min_radius + 1));
            uint32_t x = r + (rand() % (grid_size - 2 * r));
            uint32_t y = r + (rand() % (grid_size - 2 * r));
            circles_out[c].x = uint16_t(x);
            circles_out[c].y = uint16_t(y);
            circles_out[c].r = uint16_t(r);
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
                float noise = 0.9f + 0.2f * float(rand()) / float(RAND_MAX);
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
        
        // Shuffle points to remove ordering information
        for (uint32_t i = n_points - 1; i > 0; --i) {
            uint32_t j = rand() % (i + 1);
            PointData temp = points_out[i];
            points_out[i] = points_out[j];
            points_out[j] = temp;
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
        
        // Draw ground truth circles (green outline)
        for (uint32_t c = 0; c < num_circles; ++c) {
            if (circles[c].r == 0) continue;  // Skip "no circle" tokens
            int cx = int(circles[c].x * SCALE);
            int cy = int(circles[c].y * SCALE);
            int cr = int(circles[c].r * SCALE);
            bmp.draw_circle(cx, IMG_SIZE - 1 - cy, cr, 0, 200, 0, 2);
        }
        
        // Draw predicted circles (red outline, if provided)
        for (uint32_t c = 0; c < num_predicted; ++c) {
            if (predicted[c].r == 0) continue;
            int cx = int(predicted[c].x * SCALE);
            int cy = int(predicted[c].y * SCALE);
            int cr = int(predicted[c].r * SCALE);
            bmp.draw_circle(cx, IMG_SIZE - 1 - cy, cr, 200, 50, 50, 2);
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
        
        // Draw loss curve (cyan color)
        int prev_x = -1, prev_y = -1;
        for (size_t i = 0; i < losses.size(); ++i) {
            int x = MARGIN + int(float(i) / float(losses.size() - 1) * float(plot_width));
            float norm_loss = (losses[i] - min_loss) / range;
            int y = height - MARGIN - int(norm_loss * float(plot_height));
            
            // Clamp y
            y = (std::max)(MARGIN, (std::min)(height - MARGIN, y));
            
            if (prev_x >= 0) {
                bmp.draw_line(prev_x, prev_y, x, y, 0, 200, 255);
            }
            prev_x = x;
            prev_y = y;
        }
        
        // Draw points on the curve
        for (size_t i = 0; i < losses.size(); i += (std::max)(size_t(1), losses.size() / 20)) {
            int x = MARGIN + int(float(i) / float(losses.size() - 1) * float(plot_width));
            float norm_loss = (losses[i] - min_loss) / range;
            int y = height - MARGIN - int(norm_loss * float(plot_height));
            y = (std::max)(MARGIN, (std::min)(height - MARGIN, y));
            bmp.draw_point(x, y, 3, 255, 255, 0);
        }
        
        bmp.save(filename);
        // printf("  Saved loss graph: %s (min=%.4f, max=%.4f)\n", filename, min_loss + range * 0.1f, max_loss - range * 0.1f);
    }
};

// Transformer for circle detection
// Input: N_POINTS 2D points -> encoded as tokens
// Output: N_MAX_PRIMS * 3 values (x, y, r for each circle)
struct CircleDetector {
    // Hyperparameters
    uint32_t n_points;          // Number of input points
    uint32_t n_max_prims;       // Max number of output circles
    uint32_t grid_size;         // Discrete coordinate space
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t batch_size;
    uint32_t num_layers;
    
    // Computed dimensions
    uint32_t vocab_size;        // grid_size for x, y, r
    uint32_t input_seq_len;     // n_points * 2 (x, y pairs)
    uint32_t output_seq_len;    // n_max_prims * 3 (x, y, r per circle)
    uint32_t total_seq_len;     // input + output sequence
    
    // Training graph
    Graph model;
    Tensor* input_tokens = nullptr;     // (B, input_seq_len) - unused, for reference
    Tensor* full_input = nullptr;       // (B, total_seq_len) - actual transformer input
    Tensor* target_tokens = nullptr;    // (B, total_seq_len)
    Tensor* loss = nullptr;
    
    // Weight tensors
    Tensor* token_emb = nullptr;        // (vocab_size, embed_dim)
    Tensor* pos_emb = nullptr;          // (total_seq_len, embed_dim)
    Tensor* w_out = nullptr;            // (embed_dim, vocab_size)
    
    // Attention blocks
    std::vector<AttentionBlock> blocks;
    
    // Inference graph
    Graph inference;
    Tensor* inf_input = nullptr;
    Tensor* inf_logits = nullptr;
    Tensor* inf_token_emb = nullptr;
    Tensor* inf_pos_emb = nullptr;
    Tensor* inf_w_out = nullptr;
    std::vector<AttentionBlock> inf_blocks;
    
    CircleDetector(uint32_t n_points_, uint32_t n_max_prims_, uint32_t grid_size_,
                   uint32_t embed_dim_, uint32_t hidden_dim_, uint32_t batch_size_,
                   uint32_t num_layers_ = 2)
        : n_points(n_points_), n_max_prims(n_max_prims_), grid_size(grid_size_),
          embed_dim(embed_dim_), hidden_dim(hidden_dim_), batch_size(batch_size_),
          num_layers(num_layers_)
    {
        // Pad vocab_size to multiple of 16 for matmul tile alignment
        uint32_t raw_vocab = grid_size + 1;  // +1 for "no value" token (0)
        vocab_size = ((raw_vocab + 15) / 16) * 16;
        
        input_seq_len = n_points * 2;  // x, y for each point
        output_seq_len = n_max_prims * 3;  // x, y, r for each circle
        
        // Pad total sequence length to multiple of 16 for matmul tile alignment
        uint32_t raw_total = input_seq_len + output_seq_len;
        total_seq_len = ((raw_total + 15) / 16) * 16;
        
        build_training_graph();
        build_inference_graph();
    }
    
    void build_training_graph() {
        // Input: points encoded as sequence of x, y coordinates
        input_tokens = &model.tensor({batch_size, input_seq_len});
        
        // Target: circle parameters encoded as sequence of x, y, r values
        // We use teacher forcing: input is [points, shifted_targets]
        // Full sequence for transformer
        full_input = &model.tensor({batch_size, total_seq_len});
        target_tokens = &model.tensor({batch_size, total_seq_len});
        
        // Learnable parameters
        token_emb = &model.tensor({vocab_size, embed_dim}, true);
        pos_emb = &model.tensor({total_seq_len, embed_dim}, true);
        w_out = &model.tensor({embed_dim, vocab_size}, true);
        
        // Attention blocks
        blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init(model, embed_dim, hidden_dim);
        }
        
        // Forward pass
        Tensor& embedded = model.embed(*token_emb, *full_input);
        Tensor& input_with_pos = model.add_position_embedding(embedded, *pos_emb, batch_size, total_seq_len);
        
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &blocks[i].forward(model, *x);
        }
        
        Tensor& logits = model.matmul(*x, *w_out);
        loss = &model.cross_entropy_loss(logits, *target_tokens);
    }
    
    void build_inference_graph() {
        inf_input = &inference.tensor({1, total_seq_len});
        
        inf_token_emb = &inference.tensor({vocab_size, embed_dim});
        inf_pos_emb = &inference.tensor({total_seq_len, embed_dim});
        inf_w_out = &inference.tensor({embed_dim, vocab_size});
        
        inf_blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            inf_blocks[i].init(inference, embed_dim, hidden_dim);
        }
        
        Tensor& embedded = inference.embed(*inf_token_emb, *inf_input);
        Tensor& input_with_pos = inference.add_position_embedding(embedded, *inf_pos_emb, 1, total_seq_len);
        
        Tensor* x = &input_with_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &inf_blocks[i].forward(inference, *x);
        }
        
        inf_logits = &inference.matmul(*x, *inf_w_out);
    }
    
    void init_weights(uint32_t seed = 42) {
        srand(seed);
        float scale = 0.1f / sqrtf(float(embed_dim));
        token_emb->random_init(0.1f);
        pos_emb->random_init(0.1f);
        
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init_weights(scale);
        }
        
        w_out->random_init(scale);
    }
    
    void copy_weights_to_inference() {
        evk::CmdCopy(token_emb->buffer, inf_token_emb->buffer, token_emb->shape.count() * sizeof(float16_t));
        evk::CmdCopy(pos_emb->buffer, inf_pos_emb->buffer, pos_emb->shape.count() * sizeof(float16_t));
        evk::CmdCopy(w_out->buffer, inf_w_out->buffer, w_out->shape.count() * sizeof(float16_t));
        
        for (uint32_t i = 0; i < num_layers; ++i) {
            evk::CmdCopy(blocks[i].w_q->buffer, inf_blocks[i].w_q->buffer, blocks[i].w_q->shape.count() * sizeof(float16_t));
            evk::CmdCopy(blocks[i].w1->buffer, inf_blocks[i].w1->buffer, blocks[i].w1->shape.count() * sizeof(float16_t));
            evk::CmdCopy(blocks[i].w2->buffer, inf_blocks[i].w2->buffer, blocks[i].w2->shape.count() * sizeof(float16_t));
        }
        evk::Sync();
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
        
        uint16_t* full_inp = (uint16_t*)full_input->cpu();
        uint16_t* tgt = (uint16_t*)target_tokens->cpu();
        
        for (uint32_t b = 0; b < batch_size; ++b) {
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
            // For input positions, we don't care (could be anything)
            // For output positions, target is the circle token sequence
            for (uint32_t t = 0; t < input_seq_len; ++t) {
                tgt[b * total_seq_len + t] = 0;  // Ignore loss on input positions
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
        
        full_input->cpu_upload();
        target_tokens->cpu_upload();
        
        model.eval(true);
        evk::Sync();
        
        loss->cpu_download();
        float loss_val = float(loss->cpu()[0]);
        
        model.step_adam(-learning_rate);
        evk::Sync();
        
        return loss_val;
    }
    
    // Predict circles from points (autoregressive generation)
    void predict(const PointData* points, CircleData* circles_out) {
        copy_weights_to_inference();
        
        uint16_t* inp = (uint16_t*)inf_input->cpu();
        
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
            inf_input->cpu_upload();
            inference.eval(false);
            evk::Sync();
            
            inf_logits->cpu_download();
            float16_t* logits = inf_logits->cpu();
            
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

// Main function for circle detection demo
void run_circle_detection() {
    printf("\n=== Circle Detection Transformer ===\n");
    
    // Hyperparameters - start with single circle for simpler task
    constexpr uint32_t N_POINTS = 16;       // Number of input points
    constexpr uint32_t N_MAX_PRIMS = 1;     // Start with just 1 circle
    constexpr uint32_t GRID_SIZE = 64;      // Smaller discrete coordinate space
    constexpr uint32_t EMBED_DIM = 64;
    constexpr uint32_t HIDDEN_DIM = 128;
    constexpr uint32_t BATCH_SIZE = 16;
    constexpr uint32_t NUM_LAYERS = 4;
    
    printf("  Config: n_points=%u, n_max_prims=%u, grid=%u, embed=%u, hidden=%u, layers=%u\n",
           N_POINTS, N_MAX_PRIMS, GRID_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
    
    // Create dataset generator
    CircleDataset dataset(N_POINTS, N_MAX_PRIMS, GRID_SIZE);
    
    // Create model
    CircleDetector detector(N_POINTS, N_MAX_PRIMS, GRID_SIZE, EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, NUM_LAYERS);
    detector.init_weights(42);
    
    printf("  Input seq len: %u, Output seq len: %u, Total: %u\n",
           detector.input_seq_len, detector.output_seq_len, detector.total_seq_len);
    
    // Training
    const int EPOCHS = 600;
    const float LR = 0.005f;
    
    printf("  Training for %d epochs...\n", EPOCHS);
    
    std::vector<float> loss_history;
    loss_history.reserve(EPOCHS);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = detector.train_batch(dataset, LR);
        loss_history.push_back(epoch_loss);
        
        if (epoch % 20 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, epoch_loss);
            CircleDataset::save_loss_graph("loss_graph.bmp", loss_history);
        }
    }
    
    
    // Evaluation and visualization
    printf("\n  === Evaluation ===\n");
    
    // Generate test samples and visualize predictions
    const int NUM_TEST = 5;
    std::vector<CircleData> gt_circles(N_MAX_PRIMS);
    std::vector<CircleData> pred_circles(N_MAX_PRIMS);
    std::vector<PointData> points(N_POINTS);
    
    srand(12345);  // Different seed for test samples
    
    for (int t = 0; t < NUM_TEST; ++t) {
        uint32_t num_gt = dataset.generate_sample(gt_circles.data(), points.data());
        detector.predict(points.data(), pred_circles.data());
        
        // Print results
        printf("  Test %d: GT circles: ", t);
        for (uint32_t c = 0; c < num_gt; ++c) {
            printf("(%d,%d,r=%d) ", gt_circles[c].x, gt_circles[c].y, gt_circles[c].r);
        }
        printf("\n          Predicted:  ");
        for (uint32_t c = 0; c < N_MAX_PRIMS; ++c) {
            if (pred_circles[c].r > 0) {
                printf("(%d,%d,r=%d) ", pred_circles[c].x, pred_circles[c].y, pred_circles[c].r);
            }
        }
        printf("\n");
        
        // Save visualization
        char filename[64];
        snprintf(filename, sizeof(filename), "circle_test_%d.bmp", t);
        dataset.save_sample_bmp(filename, gt_circles.data(), num_gt, 
                               points.data(), N_POINTS,
                               pred_circles.data(), N_MAX_PRIMS);
        printf("          Saved: %s\n", filename);
    }
}
