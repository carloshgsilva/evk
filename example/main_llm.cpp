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

void main_llm() {
    printf("=== main_llm ===\n");
    run_next_token_prediction_attention();
}
