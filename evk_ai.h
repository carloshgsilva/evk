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
    uint32_t values[MAX_DIMENSIONS];
    uint32_t size;

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
    Shape shape;

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

    // ReLU activation (CPU implementation for now)
    // out = max(0, in)
    void relu(Tensor& in, Tensor& out);

    // ReLU backward: grad_in = grad_out * (in > 0 ? 1 : 0)
    void relu_backward(Tensor& grad_out, Tensor& in, Tensor& grad_in);
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

    Tensor& matmul(Tensor& a, Tensor& b, uint8_t tile_m = 16, uint8_t tile_n = 16) {
        nodes.push_back(std::make_unique<Tensor>(Shape({a.shape[0], b.shape[1]})));
        Tensor& c = *nodes.back();
        c.forward_fn = [&a, &b, &c, tile_m, tile_n]() {
            evk::ai::matmul(a, b, c, false, false, false, tile_m, tile_n);
        };
        c.backward_fn = [&a, &b, &c, tile_m, tile_n]() {
            evk::ai::matmul(c.grad(), b, a.grad(), false, true, false, tile_m, tile_n);
            evk::ai::matmul(a, c.grad(), b.grad(), true, false, false, tile_m, tile_n);
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

    // eval the graph
    // if backward is true, also run the backward pass
    void eval(bool backward = false) {
        for(auto& node : nodes) {
            if (node->forward_fn) {
                node->forward_fn();
            }
        }
        if (backward) {
            // Zero parameter grad tensors before running backward so parameter grads don't accumulate
            for (auto& param : params) {
                if (param->grad_tensor) {
                    float16_t* gcpu = param->grad_tensor->cpu();
                    memset(gcpu, 0, sizeof(float16_t) * param->grad_tensor->shape.count());
                    param->grad_tensor->cpu_upload();
                }
            }

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
    // Note: uses -lr internally to match gradient direction convention from mse_loss
    void step_adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-4f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::adam(*param, param->grad(), adam_states[param], -lr, beta1, beta2, epsilon);
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

