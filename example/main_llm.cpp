#include <iostream>
#include <evk_ai.h>

#include "bmp.h"

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }


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
    const uint32_t ACTUAL_VOCAB = 30;
    
    auto char_to_token = [&](char c) -> uint16_t {
        for (uint32_t i = 0; VOCAB[i]; ++i) {
            if (VOCAB[i] == c) return uint16_t(i);
        }
        return 0;
    };

    
    const uint32_t SEQ_LEN = 32;
    const uint32_t EMBED_DIM = 64;
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t BATCH_SIZE = 4;
    
    printf("  Attention Model: vocab=%u, seq=%u, embed=%u, hidden=%u\n", 
           VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    
    Graph model;
    
    // Use 3D tensors throughout: (B, N, ...)
    Tensor& input_tokens = model.tensor({BATCH_SIZE, SEQ_LEN});
    Tensor& targets = model.tensor({BATCH_SIZE, SEQ_LEN});
    
    Tensor& token_emb = model.tensor({VOCAB_SIZE, EMBED_DIM}, true);
    Tensor& pos_emb = model.tensor({SEQ_LEN, EMBED_DIM}, true);
    
    // Attention projection weights
    Tensor& w_q = model.tensor({EMBED_DIM, EMBED_DIM}, true);
    Tensor& w_k = model.tensor({EMBED_DIM, EMBED_DIM}, true);
    Tensor& w_v = model.tensor({EMBED_DIM, EMBED_DIM}, true);
    
    // FFN weights
    Tensor& w1 = model.tensor({EMBED_DIM, HIDDEN_DIM}, true);
    Tensor& w2 = model.tensor({HIDDEN_DIM, EMBED_DIM}, true);
    Tensor& w_out = model.tensor({EMBED_DIM, VOCAB_SIZE}, true);
    
    // embed now returns (B, N, D) shape
    Tensor& embedded = model.embed(token_emb, input_tokens);
    Tensor& input_with_pos = model.add_position_embedding(embedded, pos_emb, BATCH_SIZE, SEQ_LEN);
    
    // Project to Q, K, V - all operations work with (B, N, D) directly
    Tensor& q = model.matmul(input_with_pos, w_q);
    Tensor& k = model.matmul(input_with_pos, w_k);
    Tensor& v = model.matmul(input_with_pos, w_v);
    
    // Causal self-attention - no view() needed, already (B, N, D)
    Tensor& attn_out = model.causal_attention(q, k, v);
    
    // Residual connection after attention
    Tensor& attn_residual = model.add(input_with_pos, attn_out);
    
    // FFN - matmul broadcasts (B, N, D) @ (D, H) -> (B, N, H)
    Tensor& hidden = model.matmul(attn_residual, w1);
    Tensor& hidden_relu = model.relu(hidden);
    Tensor& hidden_proj = model.matmul(hidden_relu, w2);
    Tensor& residual = model.add(attn_residual, hidden_proj);
    Tensor& logits = model.matmul(residual, w_out);
    Tensor& loss = model.cross_entropy_loss(logits, targets);
    
    srand(42);
    float scale = 0.1f / sqrtf(float(EMBED_DIM));
    token_emb.random_init(0.1f);
    pos_emb.random_init(0.1f);
    w_q.random_init(scale);
    w_k.random_init(scale);
    w_v.random_init(scale);
    w1.random_init(scale);
    w2.random_init(scale);
    w_out.random_init(scale);
    
    const int EPOCHS = 120;
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
        
        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            printf("  epoch %3d: loss = %.4f\n", epoch, avg_loss);
        }
        
        last_loss = avg_loss;
    }
    
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initial_loss, last_loss);
    
    // === Autoregressive Generation ===
    printf("\n  === Autoregressive Generation ===\n");
    
    // Create inference graph with batch size 1, using 3D tensors throughout
    Graph inference;
    
    Tensor& inf_input = inference.tensor({1, SEQ_LEN});
    
    // Copy trained weights to inference graph (share the same tensors)
    Tensor& inf_token_emb = inference.tensor({VOCAB_SIZE, EMBED_DIM});
    Tensor& inf_pos_emb = inference.tensor({SEQ_LEN, EMBED_DIM});
    Tensor& inf_w_q = inference.tensor({EMBED_DIM, EMBED_DIM});
    Tensor& inf_w_k = inference.tensor({EMBED_DIM, EMBED_DIM});
    Tensor& inf_w_v = inference.tensor({EMBED_DIM, EMBED_DIM});
    Tensor& inf_w1 = inference.tensor({EMBED_DIM, HIDDEN_DIM});
    Tensor& inf_w2 = inference.tensor({HIDDEN_DIM, EMBED_DIM});
    Tensor& inf_w_out = inference.tensor({EMBED_DIM, VOCAB_SIZE});
    
    // Copy weights from trained model
    evk::CmdCopy(token_emb.buffer, inf_token_emb.buffer, token_emb.shape.count() * sizeof(float16_t));
    evk::CmdCopy(pos_emb.buffer, inf_pos_emb.buffer, pos_emb.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w_q.buffer, inf_w_q.buffer, w_q.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w_k.buffer, inf_w_k.buffer, w_k.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w_v.buffer, inf_w_v.buffer, w_v.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w1.buffer, inf_w1.buffer, w1.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w2.buffer, inf_w2.buffer, w2.shape.count() * sizeof(float16_t));
    evk::CmdCopy(w_out.buffer, inf_w_out.buffer, w_out.shape.count() * sizeof(float16_t));
    evk::Sync();
    
    // Build inference graph - all 3D, no view() needed
    Tensor& inf_embedded = inference.embed(inf_token_emb, inf_input);
    Tensor& inf_input_pos = inference.add_position_embedding(inf_embedded, inf_pos_emb, 1, SEQ_LEN);
    
    Tensor& inf_q = inference.matmul(inf_input_pos, inf_w_q);
    Tensor& inf_k = inference.matmul(inf_input_pos, inf_w_k);
    Tensor& inf_v = inference.matmul(inf_input_pos, inf_w_v);
    
    // Causal attention - already (1, N, D)
    Tensor& inf_attn = inference.causal_attention(inf_q, inf_k, inf_v);
    
    Tensor& inf_attn_res = inference.add(inf_input_pos, inf_attn);
    Tensor& inf_hidden = inference.matmul(inf_attn_res, inf_w1);
    Tensor& inf_hidden_relu = inference.relu(inf_hidden);
    Tensor& inf_hidden_proj = inference.matmul(inf_hidden_relu, inf_w2);
    Tensor& inf_residual = inference.add(inf_attn_res, inf_hidden_proj);
    Tensor& inf_logits = inference.matmul(inf_residual, inf_w_out);
    
    // Token to char conversion
    auto token_to_char = [&](uint16_t t) -> char {
        return (t < ACTUAL_VOCAB) ? VOCAB[t] : '?';
    };
    
    // Generate from different prefixes
    const char* prefixes[] = {
        "lorem ",
        "dolor ",
        "in ",
        "qu",
        "ex ",
    };
    const int NUM_PREFIXES = sizeof(prefixes) / sizeof(prefixes[0]);
    const int GEN_LEN = 28; // Generate this many new tokens
    
    for (int p = 0; p < NUM_PREFIXES; ++p) {
        const char* prefix = prefixes[p];
        uint32_t prefix_len = (uint32_t)strlen(prefix);
        
        // Initialize sequence with prefix
        char generated[SEQ_LEN + 1];
        memset(generated, 0, sizeof(generated));
        strncpy(generated, prefix, SEQ_LEN);
        
        uint32_t cur_len = prefix_len;
        
        // Autoregressively generate tokens
        for (int g = 0; g < GEN_LEN && cur_len < SEQ_LEN; ++g) {
            // Fill input tokens
            uint16_t* inp = (uint16_t*)inf_input.cpu();
            for (uint32_t t = 0; t < SEQ_LEN; ++t) {
                inp[t] = (t < cur_len) ? char_to_token(generated[t]) : uint16_t(0);
            }
            inf_input.cpu_upload();
            
            // Forward pass (no backward)
            inference.eval(false);
            evk::Sync();
            
            // Get logits for the last position
            inf_logits.cpu_download();
            float16_t* log_ptr = inf_logits.cpu();
            
            // Find argmax at position (cur_len - 1)
            uint32_t last_pos = cur_len - 1;
            float max_val = -1e9f;
            uint16_t max_idx = 0;
            for (uint32_t v = 0; v < VOCAB_SIZE; ++v) {
                float val = float(log_ptr[last_pos * VOCAB_SIZE + v]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = uint16_t(v);
                }
            }
            
            // Append generated token
            generated[cur_len] = token_to_char(max_idx);
            cur_len++;
        }
        
        generated[cur_len] = '\0';
        printf("  \"%s\" -> \"%s\"\n", prefix, generated);
    }
    
    return last_loss;
}

void main_llm() {
    printf("=== main_llm ===\n");
    run_next_token_prediction_attention();
    // test_next_token_prediction_fnet();
    // compare_attention_vs_fnet();
}
