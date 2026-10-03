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

namespace evk::ai {
    evk::Cmd& GetCmd();
    uint64_t SubmitCmd(bool wait = true);

    namespace detail {
        class CommandScope {
        public:
            explicit CommandScope(evk::Cmd& cmd);
            ~CommandScope();
            CommandScope(const CommandScope&) = delete;
            CommandScope& operator=(const CommandScope&) = delete;

        private:
            evk::Cmd* previous_;
        };
    }

    template<typename Record>
    void WithCmd(evk::Cmd& cmd, Record&& record) {
        detail::CommandScope scope(cmd);
        record();
    }
}

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

    Shape() = default;
    Shape(std::initializer_list<uint32_t> shape_values)
        : size(uint32_t(shape_values.size())) {
        assert(shape_values.size() <= MAX_DIMENSIONS);
        std::copy(shape_values.begin(), shape_values.end(), values);
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

    friend bool operator==(const Shape& a, const Shape& b) {
        return a.size == b.size &&
               std::equal(a.values, a.values + a.size, b.values);
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

struct GradStats {
    float min_val = 1e9f;
    float max_val = -1e9f;
    float mean = 0.0f;
    float rms = 0.0f;
    uint32_t nan_count = 0;
    uint32_t inf_count = 0;
    uint32_t zero_count = 0;
    
    void print(const char* name) const {
        printf("    %s: min=%.6f, max=%.6f, mean=%.6f, rms=%.6f, nan=%u, inf=%u, zero=%u\n",
               name, min_val, max_val, mean, rms, nan_count, inf_count, zero_count);
    }
};

enum class VarianceScaleMode {
    FanIn,
    FanOut,
    FanAverage,
};

struct Tensor {
    evk::Buffer buffer;
    evk::Buffer cpu_buffer;
    std::unique_ptr<Tensor> grad_tensor;
    Shape shape = {};
    std::string name;

    std::function<void()> forward_fn;
    std::function<void()> backward_fn;

    Tensor() = default;
    Tensor(const Shape& shape) {
        this->shape = shape;
        // compute total size as product of first `count` dimensions
        uint32_t s = shape.count() * sizeof(float16_t);
        buffer = evk::CreateBuffer({
            .size = s,
            .usage = evk::BufferUsage::Storage,
        });
    }

    static Tensor alias(const Shape& shape, const evk::Buffer& existing_buffer) {
        Tensor t;
        t.shape = shape;
        t.buffer = existing_buffer;
        return t;
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
    void cpu_upload(bool submit = true) {
        cpu();
        auto& cmd = evk::ai::GetCmd();
        cmd.copy(cpu_buffer, buffer, shape.count() * sizeof(float16_t));
        cmd.barrier();
        if (submit) {
            evk::ai::SubmitCmd(true);
        }
    }
    void cpu_download(bool submit = true) {
        cpu();
        auto& cmd = evk::ai::GetCmd();
        cmd.barrier();
        cmd.copy(buffer, cpu_buffer, shape.count() * sizeof(float16_t));
        if (submit) {
            evk::ai::SubmitCmd(true);
        }
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

    // Initialize with scaled Gaussian random values.
    // stddev controls the standard deviation of the sampled weights.
    Tensor& random_init(float stddev = 0.1f) {
        float16_t* data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            // Box-Muller transform for Gaussian
            float u1 = float(rand() + 1) / float(RAND_MAX + 1);
            float u2 = float(rand()) / float(RAND_MAX);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            data[i] = float16_t(z * stddev);
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

    GradStats stats() {
        cpu_download();
        float16_t* data = cpu();
        uint32_t count = shape.count();

        GradStats stats;
        float sum = 0.0f;
        float sum_sq = 0.0f;
        uint32_t valid_count = 0;

        for (uint32_t i = 0; i < count; ++i) {
            float v = float(data[i]);
            if (std::isnan(v)) {
                stats.nan_count++;
            } else if (std::isinf(v)) {
                stats.inf_count++;
            } else {
                if (v == 0.0f) stats.zero_count++;
                stats.min_val = (std::min)(stats.min_val, v);
                stats.max_val = (std::max)(stats.max_val, v);
                sum += v;
                sum_sq += v * v;
                valid_count++;
            }
        }

        if (valid_count > 0) {
            stats.mean = sum / float(valid_count);
            stats.rms = sqrtf(sum_sq / float(valid_count));
        }

        return stats;
    }

    private:
    std::vector<float16_t> cpu_data;
};

// pure fp16 and u16 tensors machine learning library
namespace evk::ai {
    // Adam optimizer state for a single parameter tensor.
    // State stays in fp16 for bandwidth.
    // m_buffer stores the first moment normalized by RMS.
    // v_buffer stores log2(RMS(second moment)) with a positive bias so tiny RMS values survive fp16 storage.
    struct AdamState {
        evk::Buffer m_buffer;  // FP16 normalized first moment estimate
        evk::Buffer v_buffer;  // FP16 biased log2 RMS(second moment) estimate
        uint32_t t = 0;        // Timestep counter

        void init(uint32_t num_elements);
        void reset();
    };

    void initialize();
    void shutdown();

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    // Supports broadcasting one operand across batch by using zero batch stride.
    void matmul(Tensor& a, Tensor& b, Tensor& c, bool transpose_a = false,
                bool transpose_b = false, bool acc_c = false,
                uint8_t TILE_M = 80u, uint8_t TILE_N = 80u,
                Tensor* residual = nullptr, Tensor* gelu_output = nullptr);
    void matmul_weight_backward(Tensor& input, Tensor& grad_output,
                                Tensor& grad_weight);

    // Bound the shared FP16 score/query tiles to 32 KiB per workgroup.
    constexpr bool supports_fused_causal_attention(uint32_t n, uint32_t d) {
        return n > 0u && d > 0u && n % 16u == 0u && d % 16u == 0u &&
               uint64_t(n) + d + 16u <= 1024u;
    }

    // Fused causal attention for supported shapes. All buffers are FP16.
    // Q/K/V/output: (B,N,D); probabilities/grad_scores workspace: (B,N,N).
    // Backward overwrites workspace and accumulates into the Q/K/V gradients.
    void causal_attention(Tensor& q, Tensor& k, Tensor& v, Tensor& probabilities,
                          Tensor& output, float scale, uint32_t window = 0);
    void causal_attention_backward(Tensor& q, Tensor& k, Tensor& v, Tensor& probabilities,
                                   Tensor& grad_output, Tensor& grad_q, Tensor& grad_k,
                                   Tensor& grad_v, Tensor& grad_scores, float scale, uint32_t window = 0);

    // Inference-only ring KV cache: rows (1,B,D), caches (B,capacity,D).
    // Keys receive RoPE when appended; queries must already have RoPE applied.
    void attention_cache_append(Tensor& k, Tensor& v, Tensor& key_cache, Tensor& value_cache,
                                uint32_t position, float rope_base = 10000.0f);
    void cached_attention(Tensor& q, Tensor& key_cache, Tensor& value_cache,
                          Tensor& output, uint32_t position);

    // Packed last dimension [gate,value], out = silu(gate) * value.
    void swiglu(Tensor& input, Tensor& output);
    void swiglu_backward(Tensor& input, Tensor& grad_output, Tensor& grad_input);

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

    // Softmax along the last dimension, with an optional input scale applied in float before exponentiation.
    // out has the same shape as in
    void softmax(Tensor& in, Tensor& out, float input_scale = 1.0f);

    // Softmax backward with optional scale factor
    // grad_in = probs * (grad_out - dot(grad_out, probs)) * scale
    void softmax_backward(Tensor& probs, Tensor& grad_out, Tensor& grad_in, float scale_factor = 1.0f);

    // ReLU activation (GPU implementation)
    // out = max(0, in)
    void relu(Tensor& in, Tensor& out);

    // ReLU backward: grad_in = grad_out * (in > 0 ? 1 : 0)
    void relu_backward(Tensor& grad_out, Tensor& in, Tensor& grad_in);

    // Inference-oriented 2D image operations. Images use NCHW layout and
    // convolution weights use OIHW layout.
    void conv2d(Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                uint32_t stride = 1, uint32_t padding = 0);
    void max_pool2d(Tensor& input, Tensor& output,
                    uint32_t kernel_size = 2, uint32_t stride = 2);
    void upsample2d(Tensor& input, Tensor& output, uint32_t scale = 2);
    void concat_channels(Tensor& a, Tensor& b, Tensor& output);
    void nchw_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                     uint32_t height, uint32_t padded_width, uint32_t padded_height);

    // GELU activation (tanh approximation)
    // out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    void gelu(Tensor& in, Tensor& out);

    // GELU backward: grad_in += grad_out * dgelu/dx
    void gelu_backward(Tensor& grad_out, Tensor& in, Tensor& grad_in);

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

    // Greedy sampling over one sequence position.
    // logits: (B, S, V)
    // out_tokens: (B) token ids written as raw uint16 values in fp16 payload storage
    void greedy_sample(Tensor& logits, Tensor& out_tokens,
                       uint32_t position,
                       uint16_t token_base,
                       uint16_t token_count);

    // Greedy sampling over every row. The last dimension is the vocabulary.
    void greedy_sample_rows(Tensor& logits, Tensor& out_tokens,
                            uint16_t token_base, uint16_t token_count);

    // Embedding backward: accumulates gradients into embedding table
    // grad_out: (B, N, embed_dim) gradient from downstream
    // indices: (B, N) same indices used in forward
    // grad_embeddings: (vocab_size, embed_dim) gradient accumulator
    void embed_backward(Tensor& grad_out, Tensor& indices, Tensor& grad_embeddings);

    // Apply causal mask to attention scores (set future positions to -inf)
    // scores: (B, N, N) attention scores where scores[b, i, j] is query i attending to key j
    // For causal: j > i should be masked (set to -inf)
    void apply_causal_mask(Tensor& scores, uint32_t window = 0);

    // Position embedding addition: out = input + pos_emb (broadcast across batch)
    void position_add(Tensor& input, Tensor& pos_emb, Tensor& out,
                      uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim);

    // Position embedding backward
    void position_add_backward(Tensor& grad_out, Tensor& grad_input, Tensor& grad_pos,
                               uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim);

    // Packed [Q,K,V,decay,write] gated delta-rule associative memory.
    // Q/K receive RoPE and L2 normalization; state layout is [B,H,K,V].
    void gated_delta_projected_step(Tensor& projection, Tensor& positions,
                                    Tensor& state, Tensor& output,
                                    uint32_t head_count,
                                    float rope_base = 10000.0f,
                                    float decay_bias = -4.0f);
    // Optional FP16 cache, shaped like projection, for heads of dimension >= 2.
    // Forward stores normalized Q/K, prediction errors and gates; backward
    // must receive the same cache. The projection itself remains unchanged.
    void gated_delta_projected(Tensor& projection, Tensor& output,
                               Tensor& state_history, uint32_t model_dim,
                               uint32_t head_count, float rope_base = 10000.0f,
                               float decay_bias = -4.0f,
                               Tensor* prepared_projection = nullptr);
    void gated_delta_projected_backward(
        Tensor& projection, Tensor& state_history, Tensor& grad_output,
        Tensor& grad_projection, uint32_t model_dim, uint32_t head_count,
        float rope_base = 10000.0f, float decay_bias = -4.0f,
        uint32_t backward_chunk_size = 0u,
        Tensor* grad_state_boundaries = nullptr,
        Tensor* prepared_projection = nullptr);

    // Rotary position encoding over the last dimension.
    // input, out: (B, N, D), D must be even.
    void rope(Tensor& input, Tensor& out,
              uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim,
              float rotary_base = 10000.0f,
              float position_scale = 1.0f,
              float position_offset = 0.0f);

    // Rotary position encoding backward.
    void rope_backward(Tensor& grad_out, Tensor& grad_input,
                       uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim,
                       float rotary_base = 10000.0f,
                       float position_scale = 1.0f,
                       float position_offset = 0.0f);

    // In-place scale: tensor *= scale_factor
    void scale(Tensor& tensor, float scale_factor);

    // Zero out a tensor on GPU
    void zero(Tensor& tensor, bool barrier = true);

    // Sum across batch dimension: out[i] += sum_b(input[b, i])
    void sum_batch(Tensor& input, Tensor& output, uint32_t batch_count, uint32_t size_per_batch);

    // RMS Normalization forward: out = input / sqrt(mean(input^2) + eps)
    // Normalizes over the last dimension
    // input, output: same shape (*, D) where D is the dimension to normalize
    void rms_norm(Tensor& input, Tensor& output, float eps = 1e-4f);

    // RMS Normalization backward
    // Accumulates gradient into grad_input
    void rms_norm_backward(Tensor& input, Tensor& grad_out, Tensor& grad_input, float eps = 1e-4f);
}

struct Graph {
    std::vector<std::unique_ptr<Tensor>> nodes;
    std::vector<Tensor*> params;
    std::unordered_map<Tensor*, evk::ai::AdamState> adam_states;
    // Saved operator intermediates are not differentiable graph nodes.
    std::vector<std::unique_ptr<Tensor>> workspaces;

    Tensor& workspace(Shape shape) {
        workspaces.push_back(std::make_unique<Tensor>(shape));
        return *workspaces.back();
    }

    // Some operations need a reusable temp/scratch buffer
    struct Scratch {
        std::unique_ptr<Tensor> tensor;
        uint32_t capacity = 0;
    };
    std::vector<Scratch> scratch;

    Tensor& get_scratch(const Shape& shape, uint32_t slot = 0) {
        if (scratch.size() <= slot) scratch.resize(slot + 1u);
        auto& buffer = scratch[slot];
        if (!buffer.tensor || buffer.capacity < shape.count()) {
            buffer.tensor = std::make_unique<Tensor>(shape);
            buffer.capacity = shape.count();
        } else {
            buffer.tensor->shape = shape;
        }
        return *buffer.tensor;
    }

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
            auto& cmd = evk::ai::GetCmd();
            cmd.copy(a.buffer, out.buffer, a.shape.count() * sizeof(float16_t));
        };
        
        out.backward_fn = [&a, &out]() {
            // Gradient flows back unchanged (just reshape)
            auto& cmd = evk::ai::GetCmd();
            cmd.copy(out.grad().buffer, a.grad().buffer, a.shape.count() * sizeof(float16_t));
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
        c.name = "matmul";
        c.forward_fn = [&a, &b, &c, tile_m, tile_n]() {
            evk::ai::matmul(a, b, c, false, false, false, tile_m, tile_n);
        };
        c.backward_fn = [this, &a, &b, &c, tile_m, tile_n]() {
            // grad_a = grad_c @ b^T
            // For 3D @ 2D broadcast case: grad_c is (B,M,N), b is (K,N), grad_a is (B,M,K)
            evk::ai::matmul(c.grad(), b, a.grad(), false, true, true, tile_m, tile_n);
            
            // grad_b = a^T @ grad_c
            // For 3D @ 2D broadcast case: a is (B,M,K), grad_c is (B,M,N), grad_b is (K,N)
            // Need to sum across batch dimension
            if (a.shape.rank() == 3 && b.shape.rank() == 2) {
                uint32_t B = a.shape[0];
                uint32_t K = a.shape[2];
                uint32_t N = b.shape[1];
                if ((a.shape[1] % 16u) == 0u && (K % 16u) == 0u &&
                    (N % 16u) == 0u) {
                    evk::ai::matmul_weight_backward(a, c.grad(), b.grad());
                } else {
                    Tensor& temp_grad = get_scratch(Shape({B, K, N}));
                    evk::ai::matmul(a, c.grad(), temp_grad,
                                    true, false, false, tile_m, tile_n);
                    evk::ai::sum_batch(temp_grad, b.grad(), B, K * N);
                }
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
            evk::ai::add(a.grad(), c.grad(), a.grad());
            evk::ai::add(b.grad(), c.grad(), b.grad());
        };

        return c;
    }

    Tensor& matmul_residual(Tensor& a, Tensor& b, Tensor& residual,
                            uint8_t tile_m = 16, uint8_t tile_n = 16) {
        assert(a.shape.rank() == 3u && b.shape.rank() == 2u);
        Shape out_shape({a.shape[0], a.shape[1], b.shape[1]});
        assert(residual.shape == out_shape);
        nodes.push_back(std::make_unique<Tensor>(out_shape));
        Tensor& out = *nodes.back();
        out.name = "matmul residual";
        out.forward_fn = [&a, &b, &residual, &out, tile_m, tile_n]() {
            evk::ai::matmul(a, b, out, false, false, false,
                            tile_m, tile_n, &residual);
        };
        out.backward_fn = [this, &a, &b, &residual, &out, tile_m, tile_n]() {
            evk::ai::add(residual.grad(), out.grad(), residual.grad());
            evk::ai::matmul(out.grad(), b, a.grad(), false, true, true,
                            tile_m, tile_n);
            evk::ai::matmul_weight_backward(a, out.grad(), b.grad());
        };
        return out;
    }

    // Fuses the forward GELU into the matmul dispatch while retaining the FP16
    // pre-activation tensor needed by the exact backward pass.
    Tensor& matmul_gelu(Tensor& a, Tensor& b,
                        uint8_t tile_m = 16, uint8_t tile_n = 16) {
        assert(a.shape.rank() == 3u && b.shape.rank() == 2u);
        Shape out_shape({a.shape[0], a.shape[1], b.shape[1]});
        nodes.push_back(std::make_unique<Tensor>(out_shape));
        Tensor& preactivation = *nodes.back();
        preactivation.name = "matmul GELU";
        nodes.push_back(std::make_unique<Tensor>(out_shape));
        Tensor& out = *nodes.back();
        out.name = "GELU";
        preactivation.forward_fn = [&a, &b, &preactivation, &out,
                                    tile_m, tile_n]() {
            evk::ai::matmul(a, b, preactivation, false, false, false,
                            tile_m, tile_n, nullptr, &out);
        };
        preactivation.backward_fn = [this, &a, &b, &preactivation,
                                     tile_m, tile_n]() {
            evk::ai::matmul(preactivation.grad(), b, a.grad(), false, true, true,
                            tile_m, tile_n);
            evk::ai::matmul_weight_backward(
                a, preactivation.grad(), b.grad());
        };
        out.backward_fn = [&preactivation, &out]() {
            evk::ai::gelu_backward(out.grad(), preactivation,
                                   preactivation.grad());
        };
        return out;
    }

    Tensor& mse_loss(Tensor& predicted, Tensor& target) {
        // Ensure predicted and target have identical shapes
        assert(predicted.shape.rank() == target.shape.rank() && "mse_loss: predicted and target must have the same rank");
        for (uint32_t i = 0; i < predicted.shape.rank(); ++i) {
            assert(predicted.shape[i] == target.shape[i] && "mse_loss: predicted and target dimensions must match");
        }

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

    // 2D image operators are forward-only. They are kept generic here while
    // model-specific graph assembly belongs to the model integration.
    Tensor& conv2d(Tensor& input, Tensor& weight, Tensor& bias,
                   uint32_t stride = 1, uint32_t padding = 0) {
        assert(input.shape.rank() == 4 && "conv2d input must be NCHW");
        assert(weight.shape.rank() == 4 && "conv2d weight must be OIHW");
        assert(bias.shape.rank() == 1 && "conv2d bias must be one-dimensional");
        assert(stride > 0);
        assert(input.shape[1] == weight.shape[1]);
        assert(bias.shape[0] == weight.shape[0]);
        assert(input.shape[2] + 2u * padding >= weight.shape[2]);
        assert(input.shape[3] + 2u * padding >= weight.shape[3]);

        uint32_t output_height =
            (input.shape[2] + 2u * padding - weight.shape[2]) / stride + 1u;
        uint32_t output_width =
            (input.shape[3] + 2u * padding - weight.shape[3]) / stride + 1u;
        nodes.push_back(std::make_unique<Tensor>(Shape({
            input.shape[0], weight.shape[0], output_height, output_width
        })));
        Tensor& output = *nodes.back();
        output.forward_fn = [&input, &weight, &bias, &output, stride, padding]() {
            evk::ai::conv2d(input, weight, bias, output, stride, padding);
        };
        return output;
    }

    Tensor& max_pool2d(Tensor& input, uint32_t kernel_size = 2, uint32_t stride = 2) {
        assert(input.shape.rank() == 4 && "max_pool2d input must be NCHW");
        assert(kernel_size > 0 && stride > 0);
        assert(input.shape[2] >= kernel_size && input.shape[3] >= kernel_size);

        uint32_t output_height = (input.shape[2] - kernel_size) / stride + 1u;
        uint32_t output_width = (input.shape[3] - kernel_size) / stride + 1u;
        nodes.push_back(std::make_unique<Tensor>(Shape({
            input.shape[0], input.shape[1], output_height, output_width
        })));
        Tensor& output = *nodes.back();
        output.forward_fn = [&input, &output, kernel_size, stride]() {
            evk::ai::max_pool2d(input, output, kernel_size, stride);
        };
        return output;
    }

    Tensor& upsample2d(Tensor& input, uint32_t scale = 2) {
        assert(input.shape.rank() == 4 && "upsample2d input must be NCHW");
        assert(scale > 0);

        nodes.push_back(std::make_unique<Tensor>(Shape({
            input.shape[0], input.shape[1], input.shape[2] * scale, input.shape[3] * scale
        })));
        Tensor& output = *nodes.back();
        output.forward_fn = [&input, &output, scale]() {
            evk::ai::upsample2d(input, output, scale);
        };
        return output;
    }

    Tensor& concat(Tensor& a, Tensor& b) {
        assert(a.shape.rank() == 4 && b.shape.rank() == 4 &&
               "concat inputs must be NCHW");
        assert(a.shape[0] == b.shape[0]);
        assert(a.shape[2] == b.shape[2] && a.shape[3] == b.shape[3]);

        nodes.push_back(std::make_unique<Tensor>(Shape({
            a.shape[0], a.shape[1] + b.shape[1], a.shape[2], a.shape[3]
        })));
        Tensor& output = *nodes.back();
        output.forward_fn = [&a, &b, &output]() {
            evk::ai::concat_channels(a, b, output);
        };
        return output;
    }

    Tensor& swiglu(Tensor& input) {
        Shape shape = input.shape;
        assert(shape[-1] % 2u == 0u);
        shape.values[shape.rank() - 1u] /= 2u;
        Tensor& out = tensor(shape);
        out.forward_fn = [&input, &out]() { evk::ai::swiglu(input, out); };
        out.backward_fn = [&input, &out]() {
            evk::ai::swiglu_backward(input, out.grad(), input.grad());
        };
        return out;
    }

    // Same operations as swiglu -> matmul_residual, but dHidden is temporary.
    // Reserve scratch during construction; each callback fully consumes it.
    Tensor& swiglu_matmul_residual(Tensor& packed, Tensor& weight, Tensor& residual) {
        assert(packed.shape.rank() == 3u && weight.shape.rank() == 2u);
        assert(packed.shape[2] == 2u * weight.shape[0]);
        assert(residual.shape == Shape({packed.shape[0], packed.shape[1], weight.shape[1]}));
        Shape hidden_shape({packed.shape[0], packed.shape[1], weight.shape[0]});
        Tensor& hidden = workspace(hidden_shape);
        Tensor& out = tensor(residual.shape);
        get_scratch(hidden_shape);
        out.forward_fn = [&packed, &weight, &residual, &hidden, &out]() {
            evk::ai::swiglu(packed, hidden);
            evk::ai::matmul(hidden, weight, out, false, false, false, 16, 16, &residual);
        };
        out.backward_fn = [this, &packed, &weight, &residual, &hidden, &out, hidden_shape]() {
            Tensor& grad_hidden = get_scratch(hidden_shape);
            evk::ai::add(residual.grad(), out.grad(), residual.grad());
            evk::ai::matmul(out.grad(), weight, grad_hidden, false, true, false, 16, 16);
            evk::ai::matmul_weight_backward(hidden, out.grad(), weight.grad());
            evk::ai::swiglu_backward(packed, grad_hidden, packed.grad());
        };
        return out;
    }

    // Save activations, but share both intermediate gradients across layers.
    Tensor& swiglu_ffn(Tensor& input, Tensor& weight_in, Tensor& weight_out,
                       Tensor& residual, uint8_t tile_n = 16, float rms_epsilon = 0.0f) {
        assert(rms_epsilon >= 0.0f);
        assert(input.shape.rank() == 3u && weight_in.shape.rank() == 2u && weight_out.shape.rank() == 2u);
        assert(weight_in.shape[1] == 2u * weight_out.shape[0]);
        Shape packed_shape({input.shape[0], input.shape[1], weight_in.shape[1]});
        Shape hidden_shape({input.shape[0], input.shape[1], weight_out.shape[0]});
        assert(residual.shape == Shape({input.shape[0], input.shape[1], weight_out.shape[1]}));
        Tensor& packed = workspace(packed_shape);
        Tensor& hidden = workspace(hidden_shape);
        Tensor& out = tensor(residual.shape);
        // A positive epsilon enables input RMSNorm; zero preserves the unnormalized API.
        Tensor& projected_input = rms_epsilon > 0.0f ? workspace(input.shape) : input;
        get_scratch(hidden_shape);
        get_scratch(packed_shape, 1);
        if (rms_epsilon > 0.0f) get_scratch(input.shape);
        out.forward_fn = [&input, &projected_input, &weight_in, &weight_out, &residual, &packed, &hidden, &out, tile_n, rms_epsilon]() {
            if (rms_epsilon > 0.0f) evk::ai::rms_norm(input, projected_input, rms_epsilon);
            evk::ai::matmul(projected_input, weight_in, packed, false, false, false, 16, tile_n);
            evk::ai::swiglu(packed, hidden);
            evk::ai::matmul(hidden, weight_out, out, false, false, false, 16, 16, &residual);
        };
        out.backward_fn = [this, &input, &projected_input, &weight_in, &weight_out, &residual, &packed, &hidden, &out, tile_n, rms_epsilon]() {
            Tensor& grad_hidden = get_scratch(hidden.shape);
            Tensor& grad_packed = get_scratch(packed.shape, 1);
            evk::ai::zero(grad_packed);
            evk::ai::add(residual.grad(), out.grad(), residual.grad());
            evk::ai::matmul(out.grad(), weight_out, grad_hidden, false, true, false, 16, 16);
            evk::ai::matmul_weight_backward(hidden, out.grad(), weight_out.grad());
            evk::ai::swiglu_backward(packed, grad_hidden, grad_packed);
            // dHidden is dead, so its slot can now hold the normalization gradient.
            Tensor& grad_input = rms_epsilon > 0.0f ? get_scratch(input.shape) : input.grad();
            evk::ai::matmul(grad_packed, weight_in, grad_input, false, true, rms_epsilon <= 0.0f, 16, tile_n);
            evk::ai::matmul_weight_backward(projected_input, grad_packed, weight_in.grad());
            if (rms_epsilon > 0.0f) evk::ai::rms_norm_backward(input, grad_input, input.grad(), rms_epsilon);
        };
        return out;
    }

    Tensor& gelu(Tensor& a) {
        nodes.push_back(std::make_unique<Tensor>(a.shape));
        Tensor& out = *nodes.back();

        out.forward_fn = [&a, &out]() {
            evk::ai::gelu(a, out);
        };

        out.backward_fn = [&a, &out]() {
            evk::ai::gelu_backward(out.grad(), a, a.grad());
        };

        return out;
    }

    // Elementwise scale: out = a * factor
    // Implemented as a dedicated graph op so that deep residual stacks can
    // keep activations in a healthy range while still supporting autograd.
    Tensor& scale(Tensor& a, float factor) {
        nodes.push_back(std::make_unique<Tensor>(a.shape));
        Tensor& out = *nodes.back();

        out.forward_fn = [&a, &out, factor]() {
            // out = a * factor
            auto& cmd = evk::ai::GetCmd();
            cmd.copy(a.buffer, out.buffer, a.shape.count() * sizeof(float16_t));
            evk::ai::scale(out, factor);
        };

        out.backward_fn = [&a, &out, factor]() {
            // grad_a += grad_out * factor
            // We can safely reuse out.grad() as a temporary since all
            // consumers of out have already run their backward passes by
            // the time this executes (reverse graph order).
            evk::ai::scale(out.grad(), factor);
            evk::ai::add(a.grad(), out.grad(), a.grad());
        };

        return out;
    }

    // RMSNorm over the last dimension.
    // input: (B, N, D)
    // out = input / rms, where rms = sqrt(mean(x^2) + eps) per (b, n).
    Tensor& rms_norm(Tensor& input, float eps = 1e-4f) {  // Larger eps for fp16 stability
        assert(input.shape.rank() >= 2 && "rms_norm expects at least 2D tensor");

        nodes.push_back(std::make_unique<Tensor>(input.shape));
        Tensor& out = *nodes.back();

        out.forward_fn = [&input, &out, eps]() {
            evk::ai::rms_norm(input, out, eps);
        };

        out.backward_fn = [&input, &out, eps]() {
            evk::ai::rms_norm_backward(input, out.grad(), input.grad(), eps);
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

    Tensor& gated_delta_projected(Tensor& projection,
                                  uint32_t model_dim,
                                  uint32_t head_count,
                                  float rope_base = 10000.0f,
                                  float decay_bias = -4.0f,
                                  uint32_t backward_chunk_size = 0u) {
        assert(projection.shape.rank() == 3u);
        uint32_t batch_size = projection.shape[0];
        uint32_t sequence_length = projection.shape[1];
        assert(head_count > 0u && model_dim % head_count == 0u);
        uint32_t head_dim = model_dim / head_count;
        assert(projection.shape[2] == 3u * model_dim + 2u * head_count);
        nodes.push_back(std::make_unique<Tensor>(
            Shape({batch_size, sequence_length, model_dim})));
        Tensor& out = *nodes.back();
        out.name = "projected gated delta";
        nodes.push_back(std::make_unique<Tensor>(Shape(
            {batch_size, sequence_length, head_count, head_dim, head_dim})));
        Tensor& state_history = *nodes.back();
        Tensor* prepared_projection = nullptr;
        if (head_dim >= 2u) {
            nodes.push_back(std::make_unique<Tensor>(projection.shape));
            prepared_projection = nodes.back().get();
        }
        Tensor* grad_state_boundaries = nullptr;
        if (backward_chunk_size > 0u && backward_chunk_size < sequence_length) {
            uint32_t chunk_count =
                (sequence_length + backward_chunk_size - 1u) /
                backward_chunk_size;
            nodes.push_back(std::make_unique<Tensor>(Shape(
                {batch_size, chunk_count, head_count, head_dim, head_dim})));
            grad_state_boundaries = nodes.back().get();
        }
        out.forward_fn = [&projection, &out, &state_history, model_dim,
                          head_count, rope_base, decay_bias, prepared_projection]() {
            evk::ai::gated_delta_projected(
                projection, out, state_history, model_dim, head_count,
                rope_base, decay_bias, prepared_projection);
        };
        out.backward_fn = [&projection, &out, &state_history, model_dim,
                           head_count, rope_base, decay_bias,
                           backward_chunk_size, grad_state_boundaries, prepared_projection]() {
            evk::ai::gated_delta_projected_backward(
                projection, state_history, out.grad(), projection.grad(),
                model_dim, head_count, rope_base, decay_bias,
                backward_chunk_size, grad_state_boundaries, prepared_projection);
        };
        return out;
    }

    Tensor& gated_delta_projected_step(Tensor& projection,
                                       Tensor& positions,
                                       Tensor& state,
                                       uint32_t model_dim,
                                       uint32_t head_count,
                                       float rope_base = 10000.0f,
                                       float decay_bias = -4.0f) {
        Shape output_shape({projection.shape[0], projection.shape[1], model_dim});
        nodes.push_back(std::make_unique<Tensor>(output_shape));
        Tensor& out = *nodes.back();
        out.forward_fn = [&projection, &positions, &state, &out, head_count,
                          rope_base, decay_bias]() {
            evk::ai::gated_delta_projected_step(
                projection, positions, state, out, head_count,
                rope_base, decay_bias);
        };
        return out;
    }

    Tensor& rope(Tensor& input,
                 float rotary_base = 10000.0f,
                 float position_scale = 1.0f,
                 float position_offset = 0.0f) {
        assert(input.shape.rank() == 3 && "rope expects (B, N, D)");
        assert((input.shape[2] % 2u) == 0u && "rope requires an even embedding dimension");

        uint32_t batch_size = input.shape[0];
        uint32_t seq_len = input.shape[1];
        uint32_t embed_dim = input.shape[2];

        nodes.push_back(std::make_unique<Tensor>(input.shape));
        Tensor& out = *nodes.back();

        out.forward_fn = [&input, &out, batch_size, seq_len, embed_dim,
                          rotary_base, position_scale, position_offset]() {
            evk::ai::rope(input, out, batch_size, seq_len, embed_dim,
                          rotary_base, position_scale, position_offset);
        };

        out.backward_fn = [&input, &out, batch_size, seq_len, embed_dim,
                           rotary_base, position_scale, position_offset]() {
            evk::ai::rope_backward(out.grad(), input.grad(), batch_size, seq_len, embed_dim,
                                   rotary_base, position_scale, position_offset);
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
            Tensor flat_logits = Tensor::alias(Shape({B * N, V}), logits.buffer);
            Tensor flat_targets = Tensor::alias(Shape({B * N}), targets.buffer);
            
            // Compute loss and gradient into flat_grad
            evk::ai::cross_entropy_loss(flat_logits, flat_targets, flat_grad, loss);
        };
        loss.backward_fn = [&logits, &targets, &loss, &flat_grad, B, N, V]() {
            Tensor flat_logits = Tensor::alias(Shape({B * N, V}), logits.buffer);
            Tensor flat_targets = Tensor::alias(Shape({B * N}), targets.buffer);
            evk::ai::cross_entropy_loss(flat_logits, flat_targets, flat_grad, loss);
            auto& cmd = evk::ai::GetCmd();
            cmd.copy(flat_grad.buffer, logits.grad().buffer, logits.shape.count() * sizeof(float16_t));
        };
        return loss;
    }

    // Softmax along last dimension with backward pass.
    // input_scale is applied in float before exponentiation and mirrored in backward.
    Tensor& softmax(Tensor& input, float input_scale = 1.0f) {
        nodes.push_back(std::make_unique<Tensor>(input.shape));
        Tensor& out = *nodes.back();
        
        out.forward_fn = [&input, &out, input_scale]() {
            evk::ai::softmax(input, out, input_scale);
        };
        
        out.backward_fn = [&input, &out, input_scale]() {
            // Softmax backward on GPU, including the input scaling from forward.
            evk::ai::softmax_backward(out, out.grad(), input.grad(), input_scale);
        };
        
        return out;
    }

    // Causal self-attention: Q @ K^T (scaled) -> causal_mask -> softmax -> @ V
    // q: (B, N, D), k: (B, N, D), v: (B, N, D)
    // Returns: (B, N, D) attention output
    Tensor& causal_attention(Tensor& q, Tensor& k, Tensor& v, float scale = 0.0f,
                            uint32_t window = 0) {
        assert(q.shape.rank() == 3 && k.shape.rank() == 3 && v.shape.rank() == 3);
        uint32_t B = q.shape[0], N = q.shape[1], D = q.shape[2];
        float attn_scale = (scale > 0.0f) ? scale : (1.0f / std::sqrt(float(D)));
        if (evk::ai::supports_fused_causal_attention(N, D)) {
            Tensor& probs = workspace({B, N, N});
            Tensor& out = tensor(q.shape);
            get_scratch({B, N, N});
            out.forward_fn = [&q, &k, &v, &probs, &out, attn_scale, window]() {
                evk::ai::causal_attention(q, k, v, probs, out, attn_scale, window);
            };
            out.backward_fn = [this, &q, &k, &v, &probs, &out, attn_scale, window, B, N]() {
                Tensor& scores = get_scratch({B, N, N});
                evk::ai::causal_attention_backward(q, k, v, probs, out.grad(),
                    q.grad(), k.grad(), v.grad(), scores, attn_scale, window);
            };
            return out;
        }
        
        Tensor& scores = tensor({B, N, N});
        Tensor& probs = tensor({B, N, N});
        Tensor& out = tensor(q.shape);

        out.forward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale, window]() {
            // scores = Q @ K^T
            evk::ai::matmul(q, k, scores, false, true, false, 16, 16);

            // Apply causal mask
            evk::ai::apply_causal_mask(scores, window);
            
            // Softmax with input scaling fused in float before exponentiation
            evk::ai::softmax(scores, probs, attn_scale);
            
            // out = probs @ V
            evk::ai::matmul(probs, v, out, false, false, false, 16, 16);
        };
        
        out.backward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale, B, N, D]() {
            // Backward through probs @ V
            // grad_probs = grad_out @ V^T
            // grad_v += probs^T @ grad_out
            evk::ai::matmul(out.grad(), v, probs.grad(), false, true, false, 16, 16);
            evk::ai::matmul(probs, out.grad(), v.grad(), true, false, true, 16, 16);
            
            // Softmax backward with the same fused input scale from forward.
            evk::ai::softmax_backward(probs, probs.grad(), scores.grad(), attn_scale);
            
            // Backward through Q @ K^T
            // grad_q += grad_scores @ K
            // grad_k += grad_scores^T @ Q
            evk::ai::matmul(scores.grad(), k, q.grad(), false, false, true, 16, 16);
            evk::ai::matmul(scores.grad(), q, k.grad(), true, false, true, 16, 16);
        };
        
        return out;
    }

    // The unrotated projection is not needed by either backward operation.
    // Positive rms_epsilon enables input RMSNorm; zero leaves the input unchanged.
    Tensor& matmul_rope(Tensor& input, Tensor& weight, float base = 10000.0f,
                        float rms_epsilon = 0.0f, uint8_t tile_n = 16) {
        assert(rms_epsilon >= 0.0f);
        assert(input.shape.rank() == 3u && weight.shape.rank() == 2u);
        Shape shape({input.shape[0], input.shape[1], weight.shape[1]});
        Tensor& out = tensor(shape);
        Tensor& projected_input = rms_epsilon > 0.0f ? workspace(input.shape) : input;
        get_scratch(shape);
        if (rms_epsilon > 0.0f) get_scratch(input.shape, 1);
        out.forward_fn = [&input, &projected_input, &weight, &out, base, shape, rms_epsilon, tile_n]() {
            if (rms_epsilon > 0.0f) evk::ai::rms_norm(input, projected_input, rms_epsilon);
            evk::ai::matmul(projected_input, weight, out, false, false, false, 16, tile_n);
            evk::ai::rope(out, out, shape[0], shape[1], shape[2], base);
        };
        out.backward_fn = [this, &input, &projected_input, &weight, &out, base, shape, rms_epsilon, tile_n]() {
            Tensor& grad = get_scratch(shape);
            evk::ai::zero(grad);
            evk::ai::rope_backward(out.grad(), grad, shape[0], shape[1], shape[2], base);
            Tensor& grad_input = rms_epsilon > 0.0f ? get_scratch(input.shape, 1) : input.grad();
            evk::ai::matmul(grad, weight, grad_input, false, true, rms_epsilon <= 0.0f, 16, tile_n);
            evk::ai::matmul_weight_backward(projected_input, grad, weight.grad());
            if (rms_epsilon > 0.0f) evk::ai::rms_norm_backward(input, grad_input, input.grad(), rms_epsilon);
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
    // submit/wait control command buffer submission for batching
    void eval(bool backward = false, bool submit = true, bool wait = true,
              bool profile = false) {
        evk::ai::GetCmd();

        // Zero gradients BEFORE running forward when doing a backward pass.
        if (backward) {
            for (auto& node : nodes) {
                evk::ai::zero(node->grad(), false);
            }
            evk::ai::GetCmd().computeBarrier();
        }

        for(auto& node : nodes) {
            if (node->forward_fn) {
                if (profile && !node->name.empty()) {
                    evk::ai::GetCmd().timestamp(node->name.c_str(), node->forward_fn);
                } else {
                    node->forward_fn();
                }
            }
        }

        if (backward) {
            // Run backward functions in reverse node order so intermediate
            // operators (e.g. matmul) can populate grads for parameters.
            for (int i = int(nodes.size()) - 1; i >= 0; --i) {
                auto& node = nodes[i];
                if (node->backward_fn) {
                    if (profile && !node->name.empty()) {
                        std::string backward_name = node->name + " backward";
                        evk::ai::GetCmd().timestamp(backward_name.c_str(), node->backward_fn);
                    } else {
                        node->backward_fn();
                    }
                }
            }
        }

        if (submit) {
            evk::ai::SubmitCmd(wait);
        }
    }

    // apply the gradient update using SGD
    void step(float lr = 0.001f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::sgd(*param, param->grad(), lr);
        }
    }

    // apply the gradient update using Adam optimizer
    // Uses fp16-appropriate epsilon (default 1e-4)
    void step_adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-4f) {
        for(auto& param : params) {
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
