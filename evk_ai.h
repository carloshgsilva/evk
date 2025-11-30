#pragma once

#include <evk.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

struct float16_t {
    uint16_t value;

    // Default constructor
    float16_t() = default;

    // Constructor from float32
    float16_t(float f) {
        value = float_to_float16(f);
    }

    // Copy constructor
    float16_t(const float16_t& other) = default;

    // Assignment operator
    float16_t& operator=(const float16_t& other) = default;

    // Conversion operator to float32
    operator float() const {
        return float16_to_float(value);
    }

    // Arithmetic operators
    float16_t& operator+=(const float16_t& other) {
        *this = float16_t(float(*this) + float(other));
        return *this;
    }

    float16_t& operator/=(const float16_t& other) {
        *this = float16_t(float(*this) / float(other));
        return *this;
    }

    // Static conversion functions
    static uint16_t float_to_float16(float f) {
        // Accurate FP32 -> FP16 conversion with correct handling of
        // normals, subnormals, infinities and NaNs and round-to-nearest-even.
        uint32_t fbits;
        std::memcpy(&fbits, &f, sizeof(fbits));

        uint32_t sign = (fbits >> 31) & 0x1;
        int32_t exp = int32_t((fbits >> 23) & 0xFF) - 127;
        uint32_t mant = fbits & 0x7FFFFF;

        uint16_t hsign = uint16_t(sign << 15);

        // Handle NaN/Infinity
        if (((fbits >> 23) & 0xFF) == 0xFF) {
            if (mant == 0) {
                // Infinity
                return hsign | 0x7C00u;
            } else {
                // NaN: preserve payload (at least one bit set in mantissa)
                uint16_t payload = uint16_t((mant >> 13) & 0x3FFu);
                if (payload == 0) payload = 1; // ensure it's NaN, not Inf
                return hsign | 0x7C00u | payload;
            }
        }

        // Normalized range for FP16 exponent is [-14, +15]
        if (exp > 15) {
            // Overflow -> infinity
            return hsign | 0x7C00u;
        } else if (exp >= -14) {
            // Normalized half-precision number
            uint16_t hexp = uint16_t(exp + 15);
            // Round mantissa from 23->10 bits, round-to-nearest-even
            uint32_t mant_rounded = mant >> 13;
            uint32_t round_bits = mant & 0x1FFFu; // bits we discarded
            // Round to nearest, ties to even
            if (round_bits > 0x1000u || (round_bits == 0x1000u && (mant_rounded & 1u))) {
                ++mant_rounded;
                if (mant_rounded == 0x400u) { // mantissa overflow -> increment exponent
                    mant_rounded = 0;
                    ++hexp;
                    if (hexp == 0x1Fu) { // overflow to infinity
                        return hsign | 0x7C00u;
                    }
                }
            }
            return hsign | uint16_t(hexp << 10) | uint16_t(mant_rounded & 0x3FFu);
        } else {
            // Value too small to be represented as a normalized half.
            // It may become a subnormal half or zero.
            if (exp < -24) {
                // Underflow to signed zero
                return hsign;
            }

            // Convert to subnormal half. Add implicit leading 1 to mantissa
            mant |= 0x800000u; // restore implicit 1
            int shift = (-14 - exp);
            // shift = number of bits we need to right-shift mantissa to fit into 10 bits
            uint32_t mant_sub = mant >> (13 + shift);

            // Rounding for subnormals: look at the bit right below kept bits
            uint32_t round_bit = (mant >> (12 + shift)) & 1u;
            if (round_bit) {
                ++mant_sub;
            }

            return hsign | uint16_t(mant_sub & 0x3FFu);
        }
    }

    static float float16_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1u;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x3FFu;

        // Handle zero and subnormal
        if (exp == 0u) {
            if (mant == 0u) {
                return sign ? -0.0f : 0.0f;
            } else {
                // Subnormal half -> convert using ldexp for correctness
                float value = std::ldexp((float)mant, -24); // mant * 2^-24
                return sign ? -value : value;
            }
        }

        // Handle Inf/NaN
        if (exp == 0x1Fu) {
            if (mant == 0u) {
                return sign ? -INFINITY : INFINITY;
            } else {
                // Build a float NaN preserving payload in the high bits
                uint32_t fbits = (sign << 31) | (0xFFu << 23) | (mant << 13);
                float out;
                std::memcpy(&out, &fbits, sizeof(out));
                return out;
            }
        }

        // Normalized number
        int32_t new_exp = int32_t(exp) - 15 + 127;
        uint32_t fbits = (sign << 31) | (uint32_t(new_exp) << 23) | (mant << 13);
        float out;
        std::memcpy(&out, &fbits, sizeof(out));
        return out;
    }
};

struct Shape {
    static constexpr uint32_t MAX_DIMENSIONS = 8;
    uint32_t values[MAX_DIMENSIONS] = {};
    uint32_t size = 0;

    Shape() {
        size = 0;
    }
    Shape(std::initializer_list<uint32_t> shape_values) {
        assert(shape_values.size() <= MAX_DIMENSIONS);
        this->size = uint32_t(shape_values.size());
        int i = 0;
        for (auto it = shape_values.begin(); it != shape_values.end(); ++it) {
            this->values[i] = *it;
            ++i;
        }
    }

    uint32_t operator[] (int index) const {
        assert(index < int(size));
        if(index < 0) index = size + index;
        assert(index >= 0 && index < int(size));
        return values[index];
    }

    // return the number of dimensions/rank
    uint32_t rank() const {
        return size;
    }

    uint32_t number_of_elements(uint32_t index = 0) const {
        assert(index < size);
        uint32_t c = 1;
        for (uint32_t i = index; i < size; ++i) {
            c *= values[i];
        }
        return c;
    }

    // return the total number of elements
    uint32_t count() const {
        uint32_t c = 1;
        for (uint32_t i = 0; i < size; ++i) {
            c *= values[i];
        }
        return c;
    }

    // return the batch merged size
    // e.g. shape = (2, 3, 4, 5) and element_count = 2, then return 2 * 3
    uint32_t batch_size(uint32_t element_count) const {
        assert(element_count <= size);
        uint32_t b = 1;
        for(uint32_t i = 0; i < size - element_count; ++i) {
            b *= values[i];
        }
        return b;
    }
};

struct Tensor {
    evk::Buffer buffer;
    evk::Buffer cpu_buffer;
    std::unique_ptr<Tensor> grad_tensor;
    Shape shape = {};

    std::function<void()> forward_fn;
    std::function<void()> backward_fn;

    Tensor(const Shape& shape) {
        this->shape = shape;
        // compute total size as product of first `count` dimensions
        uint32_t s = shape.count() * sizeof(float16_t);
        buffer = evk::CreateBuffer({
            .size = s,
            .usage = evk::BufferUsage::Storage,
        });
    }

    // get the grad tensor
    // create it if it doesn't exist
    Tensor& grad() {
        if(!grad_tensor) {
            grad_tensor = std::make_unique<Tensor>(shape);
        }
        return *grad_tensor;
    }

    // copies data from CPU to GPU
    void cpu_upload() {
        cpu();
        evk::CmdCopy(cpu_buffer, buffer, shape.count() * sizeof(float16_t));
        evk::Sync();
    }
    void cpu_download() {
        cpu();
        evk::CmdCopy(buffer, cpu_buffer, shape.count() * sizeof(float16_t));
        evk::Sync();
    }
    float16_t* cpu() {
        if(!cpu_buffer) {
            cpu_buffer = evk::CreateBuffer({
                .size = shape.count() * sizeof(float16_t),
                .usage = evk::BufferUsage::TransferDst | evk::BufferUsage::TransferSrc,
                .memoryType = evk::MemoryType::CPU,
            });
        }
        return (float16_t*)cpu_buffer.GetPtr();
    }

    Tensor& identity(float16_t val = float16_t(1.0f)) {
        float16_t* data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            data[i] = float16_t((i % (shape[0]+1) == 0)? val : float16_t(0.0f));
        }
        cpu_upload();
        return *this;
    }
    Tensor& random(float16_t val = float16_t(0.0f)) {
        float16_t* data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            data[i] = float16_t(float(rand()) / float(RAND_MAX));
        }
        cpu_upload();
        return *this;
    }
    Tensor& fill(float16_t val = float16_t(0.0f)) {
        float16_t* data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            data[i] = val;
        }
        cpu_upload();
        return *this;
    }

    // Initialize with scaled Gaussian random values (for weight initialization)
    // Uses Box-Muller transform for Gaussian distribution
    Tensor& random_init(float scale = 0.1f) {
        float16_t* data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            // Box-Muller transform for Gaussian
            float u1 = float(rand() + 1) / float(RAND_MAX + 1);
            float u2 = float(rand()) / float(RAND_MAX);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            data[i] = float16_t(z * scale);
        }
        cpu_upload();
        return *this;
    }

    float16_t item() {
        assert(shape.count() == 1);
        cpu_download();
        float16_t* data = cpu();
        return data[0];
    }

    void print(uint32_t max_elements = 8, uint32_t max_batch = 4) {
        // Print shape header
        printf("Tensor (");
        for (uint32_t i = 0; i < shape.rank(); ++i) {
            if(i != 0) printf(", ");
            printf("%d", shape[i]);
        }
        printf("):\n");

        cpu_download();
        float16_t* data = cpu();

        // If rank < 2 just fallback to flat print (limited)
        if (shape.rank() < 2) {
            uint32_t to_show = (std::min)(shape.count(), max_elements);
            printf("[");
            for (uint32_t i = 0; i < to_show; ++i) {
                if (i) printf(", ");
                printf("%g", float(data[i]));
            }
            if (to_show < shape.count()) printf(", ...");
            printf("]\n");
            printf("]\n");
            return;
        }

        // Determine indices for last two dimensions
        uint32_t rows = shape[-2];
        uint32_t cols = shape[-1];

        // Determine batch count (product of all dims before last two)
        uint32_t batch = 1;
        if (shape.rank() > 2) batch = shape.batch_size(2);

        uint32_t show_batches = (std::min)(batch, max_batch);

        // stride between rows in contiguous memory for one matrix
        uint32_t matrix_size = rows * cols;

        for (uint32_t b = 0; b < show_batches; ++b) {
            if (b != 0) printf(",\n");
            printf("[\n");
            // offset to this batch's matrix start
            uint32_t batch_offset = b * matrix_size;

            uint32_t show_rows = (std::min)(rows, max_elements);
            for (uint32_t r = 0; r < show_rows; ++r) {
                if (r != 0) printf(",\n");
                printf(" %c[", r == 0 ? '[' : ' ');
                uint32_t row_offset = batch_offset + r * cols;

                uint32_t show_cols = (std::min)(cols, max_elements);
                for (uint32_t c = 0; c < show_cols; ++c) {
                    if (c != 0) printf(", ");
                    printf("%g", float(data[row_offset + c]));
                }
                if (show_cols < cols) printf(", ...");
                printf("]");
            }

            if (show_rows < rows) {
                printf(",\n  ...,\n  [");
                // print last row truncated
                uint32_t last_row = rows - 1;
                uint32_t last_row_offset = batch_offset + last_row * cols;
                uint32_t show_cols = (std::min)(cols, max_elements);
                for (uint32_t c = 0; c < show_cols; ++c) {
                    if (c != 0) printf(", ");
                    printf("%g", float(data[last_row_offset + c]));
                }
                if (show_cols < cols) printf(", ...");
                printf("]");
            }

            printf("]");
        }

        if (show_batches < batch) printf(",\n...,\n...\n");
        else printf("\n");

        printf("]\n");
    }

    private:
    std::vector<float16_t> cpu_data;
};

// pure fp16 and u16 tensors machine learning library
namespace evk::ai {
    // Adam optimizer state for a single parameter tensor
    // Maintains first moment (m) and second moment (v) estimates
    struct AdamState {
        evk::Buffer m_buffer;  // First moment estimate
        evk::Buffer v_buffer;  // Second moment estimate
        uint32_t t = 0;        // Timestep counter

        void init(uint32_t num_elements);
        void reset();
    };

    void initialize();
    void shutdown();

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    // Supports broadcasting one operand across batch by using zero batch stride.
    void matmul(Tensor& a, Tensor& b, Tensor& c, bool transpose_a = false, bool transpose_b = false, bool acc_c = false, uint8_t TILE_M = 80u, uint8_t TILE_N = 80u);

    // Fused Flash Attention forward (Multi-Query Attention)
    // New layout without head permutation:
    // Q, O: (B, N, D)  where D = H * Dh
    // K, V: (B, N, Dh) shared across heads
    void flash_attention(Tensor& q, Tensor& k, Tensor& v, Tensor& o);

    void flash_attention_bwd(Tensor& q, Tensor& k, Tensor& v, Tensor& o, Tensor& dO, Tensor& dQ, Tensor& dK, Tensor& dV, uint32_t heads = 0);

    // MSE Loss: (1/N) * sum(predicted - target)^2
    // Returns a scalar tensor containing the mean squared error
    void mse_loss(Tensor& predicted, Tensor& target, Tensor& predGrad, Tensor& result);

    // SGD: param = param - learning_rate * gradient
    void sgd(Tensor& param, Tensor& gradient, float learning_rate);

    // Adam: Adaptive Moment Estimation optimizer
    // Uses fp16-appropriate epsilon (default 1e-4) to avoid underflow
    void adam(Tensor& param, Tensor& gradient, AdamState& state,
              float learning_rate = 0.001f,
              float beta1 = 0.9f,
              float beta2 = 0.999f,
              float epsilon = 1e-4f);

    // Elementwise add: C = A + B
    void add(Tensor& a, Tensor& b, Tensor& c);

    // Softmax along last dimension
    // out has the same shape as in
    void softmax(Tensor& in, Tensor& out);

    // Softmax backward with optional scale factor
    // grad_in = probs * (grad_out - dot(grad_out, probs)) * scale
    void softmax_backward(Tensor& probs, Tensor& grad_out, Tensor& grad_in, float scale_factor = 1.0f);

    // ReLU activation (GPU implementation)
    // out = max(0, in)
    void relu(Tensor& in, Tensor& out);

    // ReLU backward: grad_in = grad_out * (in > 0 ? 1 : 0)
    void relu_backward(Tensor& grad_out, Tensor& in, Tensor& grad_in);

    // Cross entropy loss for classification
    // logits: (B*N, V) unnormalized log probabilities
    // targets: (B*N) target class indices stored as uint16
    //         NOTE: target=0 is the IGNORE token - positions with target=0
    //         are excluded from loss computation and gradients are zeroed
    // grad: (B*N, V) gradient output (softmax - one_hot, or zero if ignored)
    // result: scalar loss value (mean over non-ignored positions only)
    void cross_entropy_loss(Tensor& logits, Tensor& targets, Tensor& grad, Tensor& result);

    // Embedding lookup: out[i] = embeddings[indices[i]]
    // embeddings: (vocab_size, embed_dim)
    // indices: (B, N) indices stored as uint16
    // out: (B, N, embed_dim)
    void embed(Tensor& embeddings, Tensor& indices, Tensor& out);

    // Embedding backward: accumulates gradients into embedding table
    // grad_out: (B, N, embed_dim) gradient from downstream
    // indices: (B, N) same indices used in forward
    // grad_embeddings: (vocab_size, embed_dim) gradient accumulator
    void embed_backward(Tensor& grad_out, Tensor& indices, Tensor& grad_embeddings);

    // Apply causal mask to attention scores (set future positions to -inf)
    // scores: (B, N, N) attention scores where scores[b, i, j] is query i attending to key j
    // For causal: j > i should be masked (set to -inf)
    void apply_causal_mask(Tensor& scores);

    // Position embedding addition: out = input + pos_emb (broadcast across batch)
    void position_add(Tensor& input, Tensor& pos_emb, Tensor& out,
                      uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim);

    // Position embedding backward
    void position_add_backward(Tensor& grad_out, Tensor& grad_input, Tensor& grad_pos,
                               uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim);

    // In-place scale: tensor *= scale_factor
    void scale(Tensor& tensor, float scale_factor);

    // Zero out a tensor on GPU
    void zero(Tensor& tensor);

    // Sum across batch dimension: out[i] += sum_b(input[b, i])
    void sum_batch(Tensor& input, Tensor& output, uint32_t batch_count, uint32_t size_per_batch);
}

struct Graph {
    std::vector<std::unique_ptr<Tensor>> nodes;
    std::vector<Tensor*> params;
    std::unordered_map<Tensor*, evk::ai::AdamState> adam_states;

    // Some operations need a reusable temp/scratch buffer
    std::unique_ptr<Tensor> scratch;

    Tensor& tensor(Shape shape, bool param = false) {
        nodes.push_back(std::make_unique<Tensor>(shape));
        Tensor& tensor = *nodes.back();
        if(param) {
            params.push_back(&tensor);
        }
        return tensor;
    }

    // View/reshape a tensor with a different shape (same total elements)
    // This creates a new tensor node that shares the same underlying buffer
    // but has a different logical shape for operations like attention
    Tensor& view(Tensor& a, Shape new_shape) {
        assert(a.shape.count() == new_shape.count() && "view requires same number of elements");
        
        nodes.push_back(std::make_unique<Tensor>(new_shape));
        Tensor& out = *nodes.back();
        
        out.forward_fn = [&a, &out]() {
            // Just copy the buffer - shapes are already set
            evk::CmdCopy(a.buffer, out.buffer, a.shape.count() * sizeof(float16_t));
        };
        
        out.backward_fn = [&a, &out]() {
            // Gradient flows back unchanged (just reshape)
            evk::CmdCopy(out.grad().buffer, a.grad().buffer, a.shape.count() * sizeof(float16_t));
        };
        
        return out;
    }

    // Matrix multiplication with automatic shape inference
    // Supports 2D (M,K) @ (K,N) -> (M,N)
    // Supports 3D (B,M,K) @ (K,N) -> (B,M,N) with broadcast
    Tensor& matmul(Tensor& a, Tensor& b, uint8_t tile_m = 16, uint8_t tile_n = 16) {
        Shape out_shape;
        if (a.shape.rank() == 2 && b.shape.rank() == 2) {
            out_shape = Shape({a.shape[0], b.shape[1]});
        } else if (a.shape.rank() == 3 && b.shape.rank() == 2) {
            out_shape = Shape({a.shape[0], a.shape[1], b.shape[1]});
        } else if (a.shape.rank() == 2 && b.shape.rank() == 3) {
            out_shape = Shape({b.shape[0], a.shape[0], b.shape[2]});
        } else if (a.shape.rank() == 3 && b.shape.rank() == 3) {
            out_shape = Shape({a.shape[0], a.shape[1], b.shape[2]});
        } else {
            assert(false && "Unsupported matmul shapes");
        }
        
        nodes.push_back(std::make_unique<Tensor>(out_shape));
        Tensor& c = *nodes.back();
        c.forward_fn = [&a, &b, &c, tile_m, tile_n]() {
            evk::ai::matmul(a, b, c, false, false, false, tile_m, tile_n);
        };
        c.backward_fn = [&a, &b, &c, tile_m, tile_n]() {
            // grad_a = grad_c @ b^T
            // For 3D @ 2D broadcast case: grad_c is (B,M,N), b is (K,N), grad_a is (B,M,K)
            evk::ai::matmul(c.grad(), b, a.grad(), false, true, true, tile_m, tile_n);
            
            // grad_b = a^T @ grad_c
            // For 3D @ 2D broadcast case: a is (B,M,K), grad_c is (B,M,N), grad_b is (K,N)
            // Need to sum across batch dimension
            if (a.shape.rank() == 3 && b.shape.rank() == 2) {
                // Create temp 3D gradient, then sum across batches on GPU
                uint32_t B = a.shape[0];
                uint32_t K = a.shape[2];
                uint32_t N = b.shape[1];
                Tensor temp_grad({B, K, N});
                evk::ai::matmul(a, c.grad(), temp_grad, true, false, false, tile_m, tile_n);
                
                // Sum across batch dimension on GPU
                evk::ai::sum_batch(temp_grad, b.grad(), B, K * N);
            } else {
                evk::ai::matmul(a, c.grad(), b.grad(), true, false, true, tile_m, tile_n);
            }
        };
        return c;
    }

    Tensor& add(Tensor& a, Tensor& b) {
        nodes.push_back(std::make_unique<Tensor>(a.shape));
        Tensor& c = *nodes.back();

        c.forward_fn = [this, &a, &b, &c]() {
            evk::ai::add(a, b, c);
        };

        c.backward_fn = [this, &a, &b, &c]() {
            evk::CmdCopy(c.grad().buffer, a.grad().buffer, a.grad().shape.count() * sizeof(float16_t));
            evk::CmdCopy(c.grad().buffer, b.grad().buffer, b.grad().shape.count() * sizeof(float16_t));
        };

        return c;
    }

    Tensor& mse_loss(Tensor& predicted, Tensor& target) {
        nodes.push_back(std::make_unique<Tensor>(Shape({1})));
        Tensor& tensor = *nodes.back();
        tensor.forward_fn = [this, &predicted, &target, &tensor]() {
            evk::ai::mse_loss(predicted, target, predicted.grad(), tensor);
        };
        // mse_loss don't need 'backward_fn' because it's fused with forward
        return tensor;
    }

    Tensor& relu(Tensor& a) {
        nodes.push_back(std::make_unique<Tensor>(a.shape));
        Tensor& out = *nodes.back();

        out.forward_fn = [&a, &out]() {
            evk::ai::relu(a, out);
        };

        out.backward_fn = [&a, &out]() {
            evk::ai::relu_backward(out.grad(), a, a.grad());
        };

        return out;
    }

    // Add positional embeddings: out = input + pos_emb (broadcast across batch)
    // input: (B, N, embed_dim) - 3D input
    // pos_emb: (N, embed_dim) - positional embeddings (learnable parameter)
    // batch_size: number of batches in input (must match input.shape[0])
    // seq_len: sequence length (N, must match input.shape[1])
    // Returns: (B, N, embed_dim) input with position encodings added
    Tensor& add_position_embedding(Tensor& input, Tensor& pos_emb, uint32_t batch_size, uint32_t seq_len) {
        uint32_t embed_dim = pos_emb.shape[1];
        assert(input.shape.rank() == 3 && "input must be (B, N, D)");
        assert(input.shape[0] == batch_size);
        assert(input.shape[1] == seq_len);
        assert(input.shape[2] == embed_dim);
        assert(pos_emb.shape[0] == seq_len);
        
        nodes.push_back(std::make_unique<Tensor>(input.shape));
        Tensor& out = *nodes.back();
        
        out.forward_fn = [&input, &pos_emb, &out, batch_size, seq_len, embed_dim]() {
            evk::ai::position_add(input, pos_emb, out, batch_size, seq_len, embed_dim);
        };
        
        out.backward_fn = [&input, &pos_emb, &out, batch_size, seq_len, embed_dim]() {
            evk::ai::position_add_backward(out.grad(), input.grad(), pos_emb.grad(),
                                           batch_size, seq_len, embed_dim);
        };
        
        return out;
    }

    // Embedding lookup: out = embeddings[indices]
    // embeddings: (vocab_size, embed_dim) - learnable parameter
    // indices: (B, N) - token indices as uint16 (filled by user each batch)
    // Returns: (B, N, embed_dim) embedded tokens
    Tensor& embed(Tensor& embeddings, Tensor& indices) {
        assert(indices.shape.rank() == 2 && "indices must be (B, N)");
        uint32_t batch_size = indices.shape[0];
        uint32_t seq_len = indices.shape[1];
        uint32_t embed_dim = embeddings.shape[1];
        
        nodes.push_back(std::make_unique<Tensor>(Shape({batch_size, seq_len, embed_dim})));
        Tensor& out = *nodes.back();
        
        out.forward_fn = [&embeddings, &indices, &out]() {
            evk::ai::embed(embeddings, indices, out);
        };
        
        out.backward_fn = [&embeddings, &indices, &out]() {
            evk::ai::embed_backward(out.grad(), indices, embeddings.grad());
        };
        
        return out;
    }

    // Cross entropy loss for token prediction
    // logits: (B, N, vocab_size) - 3D logits
    // targets: (B, N) target indices as uint16
    //         NOTE: target=0 is the IGNORE token - positions with target=0
    //         are excluded from loss computation and produce zero gradients.
    //         Use this to mask out input positions in sequence-to-sequence tasks.
    // Returns scalar loss (mean over non-ignored positions only)
    Tensor& cross_entropy_loss(Tensor& logits, Tensor& targets) {
        assert(logits.shape.rank() == 3 && "logits must be (B, N, V)");
        assert(targets.shape.rank() == 2 && "targets must be (B, N)");
        
        uint32_t B = logits.shape[0];
        uint32_t N = logits.shape[1];
        uint32_t V = logits.shape[2];
        
        // Create a persistent flat gradient tensor for the kernel
        nodes.push_back(std::make_unique<Tensor>(Shape({B * N, V})));
        Tensor& flat_grad = *nodes.back();
        
        nodes.push_back(std::make_unique<Tensor>(Shape({1})));
        Tensor& loss = *nodes.back();
        
        // The kernel expects (B*N, V) logits and (B*N) targets
        // Since data is contiguous and layout matches, we can directly alias the buffers
        loss.forward_fn = [&logits, &targets, &loss, &flat_grad, B, N, V]() {
            // Directly use logits buffer aliased as flat (B*N, V)
            // Directly use targets buffer aliased as flat (B*N)
            Tensor flat_logits({B * N, V});
            flat_logits.buffer = logits.buffer;  // Alias, no copy
            Tensor flat_targets({B * N});
            flat_targets.buffer = targets.buffer;  // Alias, no copy
            
            // Compute loss and gradient into flat_grad
            evk::ai::cross_entropy_loss(flat_logits, flat_targets, flat_grad, loss);
        };
        loss.backward_fn = [&logits, &targets, &loss, &flat_grad, B, N, V]() {
            // Re-compute gradient (since eval() zeroes gradients before backward)
            Tensor flat_logits({B * N, V});
            flat_logits.buffer = logits.buffer;  // Alias, no copy
            Tensor flat_targets({B * N});
            flat_targets.buffer = targets.buffer;  // Alias, no copy
            Tensor dummy_loss({1});
            
            // Compute gradient into flat_grad, then copy to logits.grad()
            evk::ai::cross_entropy_loss(flat_logits, flat_targets, flat_grad, dummy_loss);
            
            // Copy flat gradient back to 3D logits gradient (same memory layout)
            evk::CmdCopy(flat_grad.buffer, logits.grad().buffer, logits.shape.count() * sizeof(float16_t));
        };
        return loss;
    }

    // Softmax along last dimension with backward pass
    Tensor& softmax(Tensor& input) {
        nodes.push_back(std::make_unique<Tensor>(input.shape));
        Tensor& out = *nodes.back();
        
        out.forward_fn = [&input, &out]() {
            evk::ai::softmax(input, out);
        };
        
        out.backward_fn = [&input, &out]() {
            // Softmax backward on GPU (scale = 1.0)
            // The shader accumulates directly into input.grad()
            evk::ai::softmax_backward(out, out.grad(), input.grad(), 1.0f);
        };
        
        return out;
    }

    // Causal self-attention: Q @ K^T (scaled) -> causal_mask -> softmax -> @ V
    // q: (B, N, D), k: (B, N, D), v: (B, N, D)
    // Returns: (B, N, D) attention output
    Tensor& causal_attention(Tensor& q, Tensor& k, Tensor& v, float scale = 0.0f) {
        assert(q.shape.rank() == 3 && k.shape.rank() == 3 && v.shape.rank() == 3);
        uint32_t B = q.shape[0], N = q.shape[1], D = q.shape[2];
        
        // Intermediate tensors stored in graph
        nodes.push_back(std::make_unique<Tensor>(Shape({B, N, N}))); // scores
        Tensor& scores = *nodes.back();
        nodes.push_back(std::make_unique<Tensor>(Shape({B, N, N}))); // probs
        Tensor& probs = *nodes.back();
        nodes.push_back(std::make_unique<Tensor>(Shape({B, N, D}))); // output
        Tensor& out = *nodes.back();
        
        float attn_scale = (scale > 0.0f) ? scale : (1.0f / std::sqrt(float(D)));
        
        out.forward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale, B, N, D]() {
            // scores = Q @ K^T
            evk::ai::matmul(q, k, scores, false, true, false, 16, 16);
            
            // Scale scores (GPU)
            evk::ai::scale(scores, attn_scale);
            
            // Apply causal mask
            evk::ai::apply_causal_mask(scores);
            
            // Softmax
            evk::ai::softmax(scores, probs);
            
            // out = probs @ V
            evk::ai::matmul(probs, v, out, false, false, false, 16, 16);
        };
        
        out.backward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale, B, N, D]() {
            // Backward through probs @ V
            // grad_probs = grad_out @ V^T
            // grad_v += probs^T @ grad_out
            evk::ai::matmul(out.grad(), v, probs.grad(), false, true, false, 16, 16);
            evk::ai::matmul(probs, out.grad(), v.grad(), true, false, true, 16, 16);
            
            // Softmax backward with attn_scale (GPU)
            evk::ai::softmax_backward(probs, probs.grad(), scores.grad(), attn_scale);
            
            // Backward through Q @ K^T
            // grad_q += grad_scores @ K
            // grad_k += grad_scores^T @ Q
            evk::ai::matmul(scores.grad(), k, q.grad(), false, false, true, 16, 16);
            evk::ai::matmul(scores.grad(), q, k.grad(), true, false, true, 16, 16);
        };
        
        return out;
    }

    // Residual connection: out = a + b
    // Same as add but with a clearer name for transformer blocks
    Tensor& residual(Tensor& a, Tensor& b) {
        return add(a, b);
    }

    // eval the graph
    // if backward is true, also run the backward pass
    void eval(bool backward = false) {
        for(auto& node : nodes) {
            if (node->forward_fn) {
                node->forward_fn();
            }
        }
        if (backward) {
            // Zero ALL gradient tensors before running backward (GPU)
            // This includes both parameters and intermediate tensors
            for (auto& node : nodes) {
                if (node->grad_tensor) {
                    evk::ai::zero(node->grad());
                }
            }
            evk::Sync();

            // Run backward functions in reverse node order so intermediate
            // operators (e.g. matmul) can populate grads for parameters.
            for (int i = int(nodes.size()) - 1; i >= 0; --i) {
                auto& node = nodes[i];
                if (node->backward_fn) {
                    node->backward_fn();
                }
            }
        }
    }

    // apply the gradient update using SGD
    void step(float lr = 0.001f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::sgd(*param, param->grad(), -lr);
        }
    }

    // apply the gradient update using Adam optimizer
    // Uses fp16-appropriate epsilon (default 1e-4)
    void step_adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-4f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::adam(*param, param->grad(), adam_states[param], lr, beta1, beta2, epsilon);
        }
    }

    // reset Adam optimizer states (useful when starting fresh training)
    void reset_adam() {
        for(auto& [param, state] : adam_states) {
            state.reset();
        }
        adam_states.clear();
    }
};

