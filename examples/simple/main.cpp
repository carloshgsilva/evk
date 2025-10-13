#include <iostream>
#include <evk.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>

#include "win_dbg.h"

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
    struct Pipelines {
        evk::Pipeline flash_attn;
        evk::Pipeline flash_attn_bwd;
        evk::Pipeline mse_loss;
        evk::Pipeline sgd;
    };
    std::unique_ptr<Pipelines> pipelines;

    struct MatMulConfig{
        uint16_t k;
        uint16_t n;
        uint8_t tile_m;
        uint8_t tile_n;
        uint8_t acc_c;
        uint8_t transpose_a: 1;
        uint8_t transpose_b: 1;
    
        operator uint64_t() const {
            return *(uint64_t*)this;
        }
    };
    
    std::unordered_map<uint64_t, evk::Pipeline> matmul_configs;
    
    evk::Pipeline get_matmul_pipeline(MatMulConfig config) {
        uint64_t key = config;
        auto it = matmul_configs.find(key);
        if (it != matmul_configs.end()) {
            return it->second;
        }
        evk::Pipeline pipeline = evk::CreatePipeline({
            .name = "matmul",
            .CS = evk::loadSpirvFile("shaders/bin/matmul_coop.comp.spv"),
            .constants = evk::Constant{
                uint32_t(config.k),
                uint32_t(config.n),
                uint32_t(config.tile_m),
                uint32_t(config.tile_n),
                uint32_t(config.acc_c),
                uint32_t(config.transpose_a),
                uint32_t(config.transpose_b),
            },
        });
        // printf("Created matmul pipeline for config: k=%d, n=%d, transpose_a=%d, sum_c=%d, tile_m=%d, tile_n=%d\n", config.k, config.n, config.transpose_a, config.sum_c, config.tile_m, config.tile_n);
        matmul_configs[key] = pipeline;
        return pipeline;
    }

    void initialize() {
        pipelines = std::make_unique<Pipelines>();
        pipelines->flash_attn = evk::CreatePipeline({
            .name = "flash_attention",
            .CS = evk::loadSpirvFile("shaders/bin/flash_attention.comp.spv"),
        });
        pipelines->flash_attn_bwd = evk::CreatePipeline({
            .name = "flash_attention_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/flash_attention_bwd.comp.spv"),
        });
        pipelines->mse_loss = evk::CreatePipeline({
            .name = "mse_loss",
            .CS = evk::loadSpirvFile("shaders/bin/mse_loss.comp.spv"),
        });
        pipelines->sgd = evk::CreatePipeline({
            .name = "sgd",
            .CS = evk::loadSpirvFile("shaders/bin/sgd.comp.spv"),
        });
    }
    void shutdown() {
        pipelines.reset();
        matmul_configs.clear();
    }

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    void matmul(Tensor& a, Tensor& b, Tensor& c, bool transpose_a = false, bool transpose_b = false, bool acc_c = false, uint8_t TILE_M = 80u, uint8_t TILE_N = 80u) {
        const uint32_t TILE_K = 16u;
        // const uint32_t TILE_M = 80u;
        // const uint32_t TILE_N = 64u;
        // printf("a.shape = (%d, %d), b.shape = (%d, %d), c.shape = (%d, %d)\n", a.shape[-2], a.shape[-1], b.shape[-2], b.shape[-1], c.shape[-2], c.shape[-1]);
        assert(a.shape[-2] % TILE_M == 0);
        assert(a.shape[-1] % TILE_K == 0);
        assert(b.shape[-2] % TILE_K == 0);
        assert(b.shape[-1] % TILE_N == 0);
        assert(c.shape[-2] % TILE_M == 0);
        assert(c.shape[-1] % TILE_N == 0);
        assert(a.shape.rank() >= 2);
        assert(b.shape.rank() >= 2);
        assert(c.shape.rank() >= 2);
        assert(a.shape.batch_size(2) == b.shape.batch_size(2));
        assert(a.shape[-1] == b.shape[-2]);

        // Determine batch size B. If tensor rank >= 3 use first dim as batch, otherwise 1.
        uint32_t batch = 1;
        if (a.shape.rank() >= 3) batch = a.shape.batch_size(2);

        // infer matrix dims: A is M x K, B is K x N
        uint32_t M = a.shape[-2];
        uint32_t K = a.shape[-1];
        uint32_t N = b.shape[-1];

        evk::CmdPush(evk::Constant{
            a.buffer.GetReference(),
            b.buffer.GetReference(),
            c.buffer.GetReference(),
        });

        uint32_t tilesCols = N / TILE_N; // columns
        uint32_t tilesRows = M / TILE_M; // rows
        evk::CmdBind(get_matmul_pipeline(MatMulConfig{
            .k = uint16_t(K),
            .n = uint16_t(N),
            .tile_m = TILE_M,
            .tile_n = TILE_N,
            .acc_c = acc_c,
            .transpose_a = transpose_a,
            .transpose_b = transpose_b,
        }));
        evk::CmdDispatch(tilesCols, tilesRows, batch);
    }

    // Fused Flash Attention forward (Multi-Query Attention)
    // New layout without head permutation:
    // Q, O: (B, N, D)  where D = H * Dh
    // K, V: (B, N, Dh) shared across heads
    void flash_attention(Tensor& q, Tensor& k, Tensor& v, Tensor& o) {
        assert(q.shape.rank() == 3);
        assert(k.shape.rank() == 3);
        assert(v.shape.rank() == 3);
        assert(o.shape.rank() == 3);

        uint32_t B = q.shape[0];
        uint32_t N = q.shape[1];
        uint32_t D = q.shape[2]; // total model dim
        assert(k.shape[0] == B && k.shape[1] == N);
        assert(v.shape[0] == B && v.shape[1] == N);
        assert(o.shape[0] == B && o.shape[1] == N && o.shape[2] == D);

        uint32_t Dh = k.shape[2];
        assert(v.shape[2] == Dh);
        assert(D % Dh == 0);
        uint32_t H = D / Dh;

        const uint32_t TILE = 32u;
        assert(D % TILE == 0); // D must be divisible by TILE
        assert(N % TILE == 0); // N must be divisible by TILE

        float scale = 1.0f / std::sqrt(float(Dh));

        evk::CmdBind(pipelines->flash_attn);
        evk::CmdPush(evk::Constant{
            q.buffer.GetRID(),
            k.buffer.GetRID(),
            v.buffer.GetRID(),
            o.buffer.GetRID(),
            B, H, N, D,
            scale
        });

        uint32_t groupX = D / TILE; // parallel Q (H*Dh = D)
        uint32_t groupY = 1u;       // sequence length N (loop inside kernel)
        uint32_t groupZ = B;        // one invocation per (b)
        evk::CmdDispatch(groupX, groupY, groupZ);
    }

    void flash_attention_bwd(Tensor& q, Tensor& k, Tensor& v, Tensor& o, Tensor& dO, Tensor& dQ, Tensor& dK, Tensor& dV, uint32_t heads = 0) {

    }

    // MSE Loss: (1/N) * sum(predicted - target)^2
    // Returns a scalar tensor containing the mean squared error
    void mse_loss(Tensor& predicted, Tensor& target, Tensor& predGrad, Tensor& result) {
        assert(predicted.shape.rank() == target.shape.rank());
        assert(result.shape.rank() == 1);
        for (uint32_t i = 0; i < predicted.shape.rank(); ++i) {
            assert(predicted.shape[i] == target.shape[i]);
        }

        uint32_t totalElements = predicted.shape.count();

        // First pass: compute partial sums with GPU shader
        evk::CmdBind(pipelines->mse_loss);
        evk::CmdPush(evk::Constant{
            predicted.buffer.GetReference(),
            target.buffer.GetReference(),
            result.buffer.GetReference(),
            predGrad.buffer.GetReference(),
            totalElements,
        });

        evk::CmdDispatch(1, 1, 1);
    }

    // SGD: param = param - learning_rate * gradient
    void sgd(Tensor& param, Tensor& gradient, float learning_rate) {
        assert(param.shape.rank() == gradient.shape.rank());
        for (uint32_t i = 0; i < param.shape.rank(); ++i) {
            assert(param.shape[i] == gradient.shape[i]);
        }

        uint32_t totalElements = param.shape.count();

        evk::CmdBind(pipelines->sgd);
        evk::CmdPush(evk::Constant{
            param.buffer.GetReference(),
            gradient.buffer.GetReference(),
            learning_rate,
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
    }
}

struct Graph {
    std::vector<std::unique_ptr<Tensor>> nodes;
    std::vector<Tensor*> params;

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

    Tensor& matmul(Tensor& a, Tensor& b) {
        nodes.push_back(std::make_unique<Tensor>(Shape({a.shape[0], b.shape[1]})));
        Tensor& tensor = *nodes.back();
        tensor.forward_fn = [this, &a, &b, &tensor]() {
            evk::ai::matmul(a, b, tensor);
        };
        tensor.backward_fn = [this, &a, &b, &tensor]() {
            evk::ai::matmul(a, b, a.grad(), false, true, true);
            evk::ai::matmul(a, b, b.grad(), true, false, true);
        };
        return tensor;
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

    // eval the graph
    // if backward is true, also run the backward pass
    void eval(bool backward = false) {
        for(auto& node : nodes) {
            if (node->forward_fn) {
                node->forward_fn();
            }
        }
        if (backward) {
            for(int i = params.size() - 1; i >= 0; --i) {
                auto& param = params[i];
                if (param->backward_fn) {
                    param->backward_fn();
                }
            }
        }
    }

    // apply the gradient update
    void step(float lr = 0.001f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::sgd(*param, param->grad(), lr);
        }
    }
};


void benchmark_matmul() {
    printf("benchmark_matmul()");
    const uint32_t SIZE = 4096u;
    for (uint32_t tile_m = 48; tile_m <= 224; tile_m += 16) {
        for (uint32_t tile_n = 48; tile_n <= 224; tile_n += 16) {
            if(tile_m != 96 || tile_n != 96) {
                // continue;
            }
            const uint32_t M = ((SIZE + tile_m - 1) / tile_m) * tile_m;
            const uint32_t N = ((SIZE + tile_n - 1) / tile_n) * tile_n;
            if(tile_m * tile_n >= 80*144){
                continue;
            }
            Tensor* a = new Tensor({M, SIZE});
            Tensor* b = new Tensor({SIZE, N});
            Tensor* c = new Tensor({M, N});
        
            a->identity(2.0f);
            b->identity(3.0f);
        
            // warm up
            for (uint32_t i = 0; i < 16; ++i) {
                evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
            }

            float min_ms = 1e9f;
            for(int it = 0; it < 1; ++it) {
                const uint32_t subIter = 32;
                evk::CmdTimestamp("matmul", [&]() {
                    for (uint32_t i = 0; i < subIter; ++i) {
                        evk::ai::matmul(*a, *b, *c, false, false, false, tile_m, tile_n);
                    }
                });
                
                evk::Sync();
                for(auto ts: evk::GetTimestamps()) {
                    min_ms = fminf(min_ms, float(ts.end - ts.start)/float(subIter));
                }
            }
            float tflops = float(2 * uint64_t(M) * uint64_t(N) * uint64_t(M)) / (min_ms / 1000.0f) / 1e12f;
            printf("matmul: %5.3fms (%7.3ftflops)", min_ms, tflops);
            printf(" M = %d, N = %d, tile_m = %d, tile_n = %d\n", M, N, tile_m, tile_n);

            delete a;
            delete b;
            delete c;
            evk::Sync();
        }
    }
}

#define TEST(expr) if(!(expr)) { printf("Test failed: " #expr " [%s:%d]\n", __FILE__, __LINE__); exit(1); }

void test_graph_backward() {
    printf("test_graph()\n");
    Graph graph;

    const uint32_t SIZE = 80u;
    Tensor& target = graph.tensor({SIZE, SIZE}).identity(1.0f);
    Tensor& a = graph.tensor({SIZE, SIZE}, true).identity(3.0f);
    // Tensor& b = graph.tensor({SIZE, SIZE}).random();
    // Tensor& c = graph.matmul(a, b);
    Tensor& loss = graph.mse_loss(a, target);

    for(int i = 0; i < 150; ++i) {
        graph.eval(true);
        graph.step(0.1f);
        // a.grad().print();
        // a.print();
        loss.print();
    }
}

void test_matmul() {
    printf("test_matmul()");
    const uint32_t SIZE = 4096;

    {
        const uint32_t SIZE = 4000u;
        Tensor a = Tensor({SIZE, SIZE});
        {
            float16_t* a_cpu = a.cpu();
            for(uint32_t i = 0; i < a.shape[0]; ++i) {
                a_cpu[i] = float(2);
            }
            a.cpu_upload();
        }
        Tensor b = Tensor({SIZE, SIZE});
        {
            float16_t* b_cpu = b.cpu();
            for(uint32_t i = 0; i < b.shape[0]; ++i) {
                b_cpu[i*b.shape[0]] = float(3);
            }
            b.cpu_upload();
        }
        Tensor c = Tensor({SIZE, SIZE});
        evk::ai::matmul(a, b, c);

        a.print();
        b.print();
        c.print();

        c.cpu_download();
        TEST(c.cpu()[0] == float16_t(SIZE * 2 * 3));
    }

    printf(" PASS\n");
}

void test_mse_loss() {
    printf("test_mse_loss()");
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
    TEST(grad_cpu[0] == float16_t(0.0f));
    TEST(grad_cpu[1] == float16_t(0.0f));
    TEST(grad_cpu[2] == float16_t(1.0f));

    printf(" PASS\n");
}

int main() {
    set_unhandled_exception_filter();

    evk::InitializeEVK({
        .applicationName = "evk_example",
        .applicationVersion = 1,
        .engineName = "evk_example_engine",
        .engineVersion = 1,
        .enableSwapchain = false,
    });

    evk::ai::initialize();

    // test_matmul();
    // test_mse_loss();
    test_graph_backward();

    // benchmark_matmul();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
