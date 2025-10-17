#include <iostream>
#include <evk.h>
#include <cmath>
#include <algorithm>
#include <functional>

#include "win_dbg.h"

#define assert(expr) if(!(expr)) { printf("Assertion failed: " #expr "\n"); throw std::runtime_error("Assertion failed"); }

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
    struct Pipelines {
        evk::Pipeline flash_attn;
        evk::Pipeline flash_attn_bwd;
        evk::Pipeline mse_loss;
        evk::Pipeline sgd;
        evk::Pipeline add;
        evk::Pipeline softmax;
        evk::Buffer flash_scratch;
        uint32_t flash_scratch_elems = 0;
    };
    std::unique_ptr<Pipelines> pipelines;

    // Generic hash combine function for creating hash keys from multiple values
    template<typename... Types>
    static uint64_t hash_combine(Types... values) {
        uint64_t hash = 14695981039346656037ULL; // FNV offset basis
        ((hash ^= std::hash<Types>{}(values),
          hash *= 1099511628211ULL), ...);
        return hash;
    }

    struct MatMulConfig{
        uint16_t m;
        uint16_t k;
        uint16_t n;
        uint8_t tile_m;
        uint8_t tile_n;
        uint8_t acc_c;
        uint8_t transpose_a;
        uint8_t transpose_b;

        operator uint64_t() const {
            return hash_combine(m, k, n, tile_m, tile_n, acc_c, transpose_a, transpose_b);
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
            .CS = evk::loadSpirvFile("shaders/bin/matmul.comp.spv"),
            .constants = evk::Constant{
                uint32_t(config.m),
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

    struct FlashConfig{
        uint32_t B;
        uint32_t H;
        uint32_t N;
        uint32_t D;
        float scale;

        operator uint64_t() const {
            return hash_combine(B, H, N, D, scale);
        }
    };
    std::unordered_map<uint64_t, evk::Pipeline> flash_configs;

    evk::Pipeline get_flash_pipeline(const FlashConfig& config) {
        uint64_t key = config;
        auto it = flash_configs.find(key);
        if (it != flash_configs.end()) {
            return it->second;
        }
        evk::Pipeline pipeline = evk::CreatePipeline({
            .name = "flash_attention",
            .CS = evk::loadSpirvFile("shaders/bin/flash_attention.comp.spv"),
            .constants = evk::Constant{
                uint32_t(config.B),
                uint32_t(config.H),
                uint32_t(config.N),
                uint32_t(config.D),
                float(config.scale),
            },
        });
        flash_configs[key] = pipeline;
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
        pipelines->add = evk::CreatePipeline({
            .name = "add",
            .CS = evk::loadSpirvFile("shaders/bin/add.comp.spv"),
        });
        pipelines->softmax = evk::CreatePipeline({
            .name = "softmax",
            .CS = evk::loadSpirvFile("shaders/bin/softmax.comp.spv"),
        });
        pipelines->flash_scratch = {};
        pipelines->flash_scratch_elems = 0;
    }
    void shutdown() {
        pipelines.reset();
        matmul_configs.clear();
        flash_configs.clear();
    }

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    void matmul(Tensor& a, Tensor& b, Tensor& c, bool transpose_a = false, bool transpose_b = false, bool acc_c = false, uint8_t TILE_M = 80u, uint8_t TILE_N = 80u) {
        const uint32_t TILE_K = 16u;

        // Basic rank and batch compatibility
        assert(a.shape.rank() >= 2);
        assert(b.shape.rank() >= 2);
        assert(c.shape.rank() >= 2);
        assert(a.shape.batch_size(2) == b.shape.batch_size(2));
        assert(a.shape.batch_size(2) == c.shape.batch_size(2));

        // Logical matrix dimensions honoring transpose flags
        // A logical dims: (M x K) if !transpose_a, else (K x M)
        // B logical dims: (K x N) if !transpose_b, else (N x K)
        uint32_t a_rows = transpose_a ? a.shape[-1] : a.shape[-2];
        uint32_t a_cols = transpose_a ? a.shape[-2] : a.shape[-1];
        uint32_t b_rows = transpose_b ? b.shape[-1] : b.shape[-2];
        uint32_t b_cols = transpose_b ? b.shape[-2] : b.shape[-1];

        // Inner dimension must match
        assert(a_cols == b_rows);

        // Output C must be (M x N) where M=a_rows, N=b_cols
        assert(c.shape[-2] == a_rows);
        assert(c.shape[-1] == b_cols);

        // Determine batch size B. If tensor rank >= 3 use batch product, otherwise 1.
        uint32_t batch = 1;
        if (a.shape.rank() >= 3) batch = a.shape.batch_size(2);

        // Tile compatibility with physical tensor strides: we operate on the logical dims
        uint32_t M = a_rows;
        uint32_t K = a_cols;
        uint32_t N = b_cols;

        // Optional tile divisibility assertions (match shader tile assumptions)
        assert(M % TILE_M == 0);
        assert(K % TILE_K == 0);
        assert(K % TILE_K == 0); // for B's K
        assert(N % TILE_N == 0);
        assert(c.shape[-2] % TILE_M == 0);
        assert(c.shape[-1] % TILE_N == 0);

        evk::CmdPush(evk::Constant{
            a.buffer.GetReference(),
            b.buffer.GetReference(),
            c.buffer.GetReference(),
        });

        uint32_t tilesCols = N / TILE_N; // columns
        uint32_t tilesRows = M / TILE_M; // rows
        evk::CmdBind(get_matmul_pipeline(MatMulConfig{
            .m = uint16_t(M),
            .k = uint16_t(K),
            .n = uint16_t(N),
            .tile_m = TILE_M,
            .tile_n = TILE_N,
            .acc_c = acc_c,
            .transpose_a = transpose_a,
            .transpose_b = transpose_b,
        }));
        evk::CmdDispatch(tilesCols, tilesRows, batch);
        evk::CmdBarrier();
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

        float scale = 1.0f / std::sqrt(float(Dh));
        // Allocate/resize scratch buffer for tiled coopmat kernel
        // Elements per tile: only Ptile (16*16)
        const uint32_t TILE_M = 16u;
        const uint32_t TILE_J = 16u;
        uint32_t tilesPerBH = (N + TILE_M - 1u) / TILE_M;
        uint32_t perTileElems = TILE_M * TILE_J;
        uint32_t totalElems = B * H * tilesPerBH * perTileElems;
        if (!pipelines->flash_scratch || pipelines->flash_scratch_elems < totalElems) {
            if (pipelines->flash_scratch) {
                pipelines->flash_scratch = {};
            }
            pipelines->flash_scratch = evk::CreateBuffer({
                .size = uint64_t(totalElems) * sizeof(float16_t),
                .usage = evk::BufferUsage::Storage,
            });
            pipelines->flash_scratch_elems = totalElems;
        }

        evk::CmdBind(get_flash_pipeline(FlashConfig{B, H, N, D, scale}));
        evk::CmdPush(evk::Constant{
            q.buffer.GetReference(),
            k.buffer.GetReference(),
            v.buffer.GetReference(),
            o.buffer.GetReference(),
            pipelines->flash_scratch.GetReference(),
        });

        // Dispatch over (tileRows, B*H)
        uint32_t groupX = 1u;
        uint32_t groupY = (N + TILE_M - 1u) / TILE_M;
        uint32_t groupZ = B * H;
        evk::CmdDispatch(groupX, groupY, groupZ);
        evk::CmdBarrier();
    }

    void flash_attention_bwd(Tensor& q, Tensor& k, Tensor& v, Tensor& o, Tensor& dO, Tensor& dQ, Tensor& dK, Tensor& dV, uint32_t heads = 0) {
        assert(q.shape.rank() == 3);
        assert(k.shape.rank() == 3);
        assert(v.shape.rank() == 3);
        assert(o.shape.rank() == 3);
        assert(dO.shape.rank() == 3);
        assert(dQ.shape.rank() == 3);
        assert(dK.shape.rank() == 3);
        assert(dV.shape.rank() == 3);

        uint32_t B = q.shape[0];
        uint32_t N = q.shape[1];
        uint32_t D = q.shape[2];
        uint32_t Dh = k.shape[2];
        assert(k.shape[0] == B && k.shape[1] == N && v.shape[0] == B && v.shape[1] == N);
        assert(o.shape[0] == B && o.shape[1] == N && o.shape[2] == D);
        assert(dO.shape[0] == B && dO.shape[1] == N && dO.shape[2] == D);
        assert(dQ.shape[0] == B && dQ.shape[1] == N && dQ.shape[2] == D);
        assert(dK.shape[0] == B && dK.shape[1] == N && dK.shape[2] == Dh);
        assert(dV.shape[0] == B && dV.shape[1] == N && dV.shape[2] == Dh);
        assert(D % Dh == 0);
        uint32_t H = D / Dh;
        if (heads != 0) {
            assert(heads == H);
        }

        float scale = 1.0f / std::sqrt(float(Dh));

        // Zero-out dQ, dK, dV before writing (since shader writes absolute values, this is optional,
        // but we keep it to avoid stale data if shapes shrink across runs)
        {
            float16_t* z;
            z = dQ.cpu(); memset(z, 0, sizeof(float16_t) * dQ.shape.count()); dQ.cpu_upload();
            z = dK.cpu(); memset(z, 0, sizeof(float16_t) * dK.shape.count()); dK.cpu_upload();
            z = dV.cpu(); memset(z, 0, sizeof(float16_t) * dV.shape.count()); dV.cpu_upload();
        }

        // Pass 1: compute dQ (mode = 0). Grid over (iTiles=N/16, B*H)
        evk::CmdBind(pipelines->flash_attn_bwd);
        struct FlashBwdPush {
            uint64_t qBuf;
            uint64_t kBuf;
            uint64_t vBuf;
            uint64_t oBuf;
            uint64_t dOBuf;
            uint64_t dQBuf;
            uint64_t dKBuf;
            uint64_t dVBuf;
            uint64_t scratchBuf;
            uint32_t B;
            uint32_t H;
            uint32_t N;
            uint32_t D;
            float scale;
            uint32_t mode;
        } push0 = {
            q.buffer.GetReference(),
            k.buffer.GetReference(),
            v.buffer.GetReference(),
            o.buffer.GetReference(),
            dO.buffer.GetReference(),
            dQ.buffer.GetReference(),
            dK.buffer.GetReference(),
            dV.buffer.GetReference(),
            pipelines->flash_scratch.GetReference(),
            B, H, N, D,
            scale,
            0u,
        };
        evk::CmdPush(push0);
        {
            const uint32_t TILE_M = 16u;
            uint32_t tilesI = (N + TILE_M - 1u) / TILE_M;
            evk::CmdDispatch(1u, tilesI, B * H);
        }
        evk::CmdBarrier();

        // Pass 2: compute dK and dV (mode = 1). Grid over (jTiles=N/16, B)
        evk::CmdBind(pipelines->flash_attn_bwd);
        FlashBwdPush push1 = {
            q.buffer.GetReference(),
            k.buffer.GetReference(),
            v.buffer.GetReference(),
            o.buffer.GetReference(),
            dO.buffer.GetReference(),
            dQ.buffer.GetReference(),
            dK.buffer.GetReference(),
            dV.buffer.GetReference(),
            pipelines->flash_scratch.GetReference(),
            B, H, N, D,
            scale,
            1u,
        };
        evk::CmdPush(push1);
        {
            const uint32_t TILE_J = 16u;
            uint32_t tilesJ = (N + TILE_J - 1u) / TILE_J;
            evk::CmdDispatch(1u, tilesJ, B);
        }
        evk::CmdBarrier();
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
        evk::CmdBarrier();
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

    // Elementwise add: C = A + B
    void add(Tensor& a, Tensor& b, Tensor& c) {
        assert(a.shape.rank() == b.shape.rank());
        assert(a.shape.rank() == c.shape.rank());
        for (uint32_t i = 0; i < a.shape.rank(); ++i) {
            assert(a.shape[i] == b.shape[i]);
            assert(a.shape[i] == c.shape[i]);
        }

        uint32_t totalElements = a.shape.count();

        // Use GPU shader pipeline to perform elementwise add
        evk::CmdBind(pipelines->add);
        evk::CmdPush(evk::Constant{
            a.buffer.GetReference(),
            b.buffer.GetReference(),
            c.buffer.GetReference(),
            totalElements,
        });
        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // Softmax along last dimension
    // out has the same shape as in
    void softmax(Tensor& in, Tensor& out) {
        assert(in.shape.rank() == out.shape.rank());
        for (uint32_t i = 0; i < in.shape.rank(); ++i) {
            assert(in.shape[i] == out.shape[i]);
        }

        uint32_t lastDim = in.shape[-1];
        uint32_t outerCount = in.shape.count() / lastDim;

        evk::CmdBind(pipelines->softmax);
        evk::CmdPush(evk::Constant{
            in.buffer.GetReference(),
            out.buffer.GetReference(),
            lastDim,
            outerCount,
        });
        evk::CmdDispatch(outerCount, 1, 1);
        evk::CmdBarrier();
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
        Tensor& c = *nodes.back();
        c.forward_fn = [this, &a, &b, &c]() {
            evk::ai::matmul(a, b, c);
        };
        c.backward_fn = [this, &a, &b, &c]() {
            evk::ai::matmul(c.grad(), b, a.grad(), false, true, false);
            evk::ai::matmul(a, c.grad(), b.grad(), true, false, false);
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
            // TODO: use add when summation grad is supported
            // evk::ai::add(a.grad(), c.grad(), a.grad());
            // evk::ai::add(b.grad(), c.grad(), b.grad());
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

    // apply the gradient update
    void step(float lr = 0.001f) {
        for(auto& param : params) {
            assert(param->grad_tensor);
            evk::ai::sgd(*param, param->grad(), -lr);
        }
    }
};


void benchmark_matmul() {
    printf("benchmark_matmul()");
    const uint32_t SIZE = 4096u;
    for (uint32_t tile_m = 48; tile_m <= 224; tile_m += 16) {
        for (uint32_t tile_n = 48; tile_n <= 224; tile_n += 16) {
            if(tile_m != 96 || tile_n != 96) {
                continue;
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
            for(int it = 0; it < 512; ++it) {
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

#define TEST(expr) if((expr)) { printf(" TEST(" #expr ") [PASS]\n"); } else { printf(" TEST(" #expr ") [FAIL] [%s:%d]\n", __FILE__, __LINE__); exit(1); }

void test_add() {
    printf("test_add()\n");
    Tensor a({4});
    Tensor b({4});
    Tensor out({4});

    float16_t* ap = a.cpu();
    float16_t* bp = b.cpu();
    for (uint32_t i = 0; i < 4; ++i) {
        ap[i] = float16_t(float(i));
        bp[i] = float16_t(float(i * 2));
    }
    a.cpu_upload();
    b.cpu_upload();

    evk::ai::add(a, b, out);

    out.cpu_download();
    float16_t* outp = out.cpu();
    for (uint32_t i = 0; i < 4; ++i) {
        TEST(outp[i] == float16_t(float(i) + float(i * 2)));
    }
}

void test_matmul() {
    printf("test_matmul()\n");

    {
        const uint32_t SIZE = 400u;
        Tensor a = Tensor({SIZE, SIZE});
        Tensor b = Tensor({SIZE, SIZE});
        Tensor c = Tensor({SIZE, SIZE});
        a.identity(2.0f);
        b.identity(2.0f);
        evk::ai::matmul(a, b, c);

        c.cpu_download();
        bool diagonal_test = true;
        bool off_diagonal_test = true;
        for(uint32_t i = 0; i < SIZE*SIZE; ++i) {
            uint32_t r = i / SIZE;
            uint32_t col = i % SIZE;
            if (r == col) {
                if(c.cpu()[i] != float16_t(4.0f)) {
                    diagonal_test = false;
                }
            } else {
                if(c.cpu()[i] != float16_t(0.0f)) {
                    off_diagonal_test = false;
                }
            }
        }
        TEST(diagonal_test);
        TEST(off_diagonal_test);
    }
}

void test_mse_loss() {
    printf("test_mse_loss()\n");
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
}

void test_flash_attention_backward() {
    printf("test_flash_attention_backward()\n");

    const uint32_t B = 1u;
    const uint32_t N = 16u; // tile-aligned
    const uint32_t Dh = 16u; // tile-aligned
    const uint32_t H = 2u;
    const uint32_t D = H * Dh;

    Tensor q({B, N, D});
    Tensor k({B, N, Dh});
    Tensor v({B, N, Dh});
    Tensor o({B, N, D});
    Tensor dO({B, N, D});
    Tensor dQ({B, N, D});
    Tensor dK({B, N, Dh});
    Tensor dV({B, N, Dh});

    // Use values such that softmax uniform (K=0), V=0 -> simplifies expectations
    {
        float16_t* qp = q.cpu();
        for (uint32_t i = 0; i < q.shape.count(); ++i) qp[i] = float16_t(0.0f);
        q.cpu_upload();
        float16_t* kp = k.cpu();
        for (uint32_t i = 0; i < k.shape.count(); ++i) kp[i] = float16_t(0.0f);
        k.cpu_upload();
        float16_t* vp = v.cpu();
        for (uint32_t i = 0; i < v.shape.count(); ++i) vp[i] = float16_t(0.0f);
        v.cpu_upload();
    }

    // dO pattern: d for each channel
    {
        float16_t* dop = dO.cpu();
        for (uint32_t d = 0; d < D; ++d) dop[d] = float16_t(float(d + 1));
        dO.cpu_upload();
    }

    evk::ai::flash_attention_bwd(q, k, v, o, dO, dQ, dK, dV);

    dQ.cpu_download();
    dK.cpu_download();
    dV.cpu_download();

    // With K=0 and V=0, softmax is uniform; dQ and dK should be ~0, dV equals row-wise mean of dO per kd across i=0..N-1
    bool ok_dQ = true, ok_dK = true, ok_dV = true;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            if (std::abs(float(dQ.cpu()[i*D + d])) > 1e-2f) ok_dQ = false;
        }
    }
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t kd = 0; kd < Dh; ++kd) {
            if (std::abs(float(dK.cpu()[j*Dh + kd])) > 1e-2f) ok_dK = false;
        }
    }
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t kd = 0; kd < Dh; ++kd) {
            float mean = 0.0f;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t h = 0; h < H; ++h) mean += float(dO.cpu()[i*D + h*Dh + kd]);
            }
            mean /= float(N);
            float got = float(dV.cpu()[j*Dh + kd]);
            if (std::abs(got - mean) > 5e-2f) ok_dV = false;
        }
    }
    TEST(ok_dQ);
    TEST(ok_dK);
    TEST(ok_dV);
}

void test_flash_attention_cmp_softmax() {
    printf("test_flash_attention_cmp_softmax()\n");
    const uint32_t B = 1u;
    const uint32_t N = 256u;   // tile aligned
    const uint32_t Dh = 64u;  // tile aligned
    const uint32_t H = 1u;
    const uint32_t D = H * Dh;

    Tensor q({B, N, D});
    Tensor k({B, N, Dh});
    Tensor v({B, N, Dh});
    Tensor o_flash({B, N, D});

    Tensor q_scaled({B, N, D});
    Tensor s({B, N, N});
    Tensor p({B, N, N});
    Tensor o_base({B, N, D});

    {
        float16_t* qp = q.cpu();
        float16_t* kp = k.cpu();
        float16_t* vp = v.cpu();
        for (uint32_t i = 0; i < N; ++i) {
            for (uint32_t d = 0; d < D; ++d) {
                qp[i * D + d] = float16_t(float((i + d) % 13) / 13.0f);
            }
        }
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t kd = 0; kd < Dh; ++kd) {
                kp[j * Dh + kd] = float16_t(float((j * 7 + kd) % 11) / 11.0f);
                vp[j * Dh + kd] = float16_t(float((j + 3 * kd) % 17) / 17.0f);
            }
        }
        q.cpu_upload();
        k.cpu_upload();
        v.cpu_upload();
    }

    {
        float scale = 1.0f / std::sqrt(float(Dh));
        float16_t* qsp = q_scaled.cpu();
        float16_t* qp = q.cpu();
        for (uint32_t idx = 0; idx < q.shape.count(); ++idx) {
            qsp[idx] = float16_t(float(qp[idx]) * scale);
        }
        q_scaled.cpu_upload();
    }

    for(int n = 0; n < 4; ++n) {
        const int ITERS = 8;
        evk::CmdTimestamp("flash_attention", [&]() {
            for(int it = 0; it < ITERS; ++it) {
                evk::ai::flash_attention(q, k, v, o_flash);
            }
        });

        evk::CmdTimestamp("attention", [&]() {
            for(int it = 0; it < ITERS; ++it) {
                evk::ai::matmul(q_scaled, k, s, false, true, false, 64, 64);   // S = (Q/√Dh) * K^T
                evk::ai::softmax(s, p);                                        // P = softmax(S) over last dim
                evk::ai::matmul(p, v, o_base, false, false, false, 64, 64);    // O = P * V
            }
        });

        o_flash.cpu_download();
        o_base.cpu_download();

        for(auto& ts : evk::GetTimestamps()) {
            printf("%s: %.4fms\n", ts.name, float(ts.end - ts.start)/float(ITERS));
        }

        bool ok = true;
        float16_t* ofp = o_flash.cpu();
        float16_t* obp = o_base.cpu();
        for (uint32_t i = 0; i < N * D; ++i) {
            float diff = std::abs(float(ofp[i]) - float(obp[i]));
            if (diff > 5e-2f) { ok = false; printf("diff[%u]: %f\n", i, diff); break; }
        }
        TEST(ok);
    }
}

void test_graph_backward() {
    printf("test_graph_backward() \n");
    Graph graph;

    const uint32_t SIZE = 80u;
    Tensor& target = graph.tensor({SIZE, SIZE}).identity(3.0f);

    Tensor& x = graph.tensor({SIZE, SIZE}).identity(1.0f);
    Tensor& w = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& b = graph.tensor({SIZE, SIZE}, true).identity(1.0f);
    Tensor& mul = graph.matmul(w, x);
    Tensor& y = graph.add(mul, b);
    Tensor& loss = graph.mse_loss(y, target);

    float last_loss = 1e9f;
    bool loss_test = true;
    for(int i = 0; i < 100; ++i) {
        graph.eval(true);
        graph.step(0.05f);
        float loss_val = float(loss.item());
        if(loss_val > last_loss) {
            loss_test = false;
        }
        last_loss = loss_val;
    }
    TEST(loss_test);
}

void test_attn_graph() {
    printf("test_attn_graph() \n");
    Graph graph;
    
}

void test_softmax() {
    printf("test_softmax()\n");
    // Shape (B, N, Dlast) -> softmax over Dlast
    const uint32_t B = 2;
    const uint32_t N = 3;
    const uint32_t Dlast = 5;
    Tensor in({B, N, Dlast});
    Tensor out({B, N, Dlast});

    // Fill deterministic values per row to validate probabilities sum to 1
    float16_t* ip = in.cpu();
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t d = 0; d < Dlast; ++d) {
                uint32_t idx = b * (N * Dlast) + n * Dlast + d;
                ip[idx] = float16_t(float(d - 2)); // centered range [-2..2]
            }
        }
    }
    in.cpu_upload();

    evk::ai::softmax(in, out);
    out.cpu_download();
    float16_t* op = out.cpu();

    // Check each row sums to ~1
    bool ok = true;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < Dlast; ++d) {
                uint32_t idx = b * (N * Dlast) + n * Dlast + d;
                sum += float(op[idx]);
            }
            if (std::abs(sum - 1.0f) > 5e-3f) ok = false;
        }
    }
    TEST(ok);
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

    // test_add();
    // test_matmul();
    // test_mse_loss();
    // test_flash_attention_backward();
    test_flash_attention_cmp_softmax();
    // test_softmax();
    // test_graph_backward();

    // benchmark_matmul();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
