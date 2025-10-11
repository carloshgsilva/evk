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

    // Static conversion functions
    static uint16_t float_to_float16(float f) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);

        // Extract components
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if (exponent == 0xFF) {
            // Infinity or NaN
            if (mantissa == 0) {
                // Infinity
                return (sign << 15) | 0x7C00;
            } else {
                // NaN - preserve the first 9 bits of mantissa
                return (sign << 15) | 0x7C00 | ((mantissa >> 13) & 0x3FF);
            }
        }

        if (exponent == 0) {
            // Zero or subnormal
            if (mantissa == 0) {
                return sign << 15;
            }
            // Subnormal numbers - handled by adjusting exponent
        }

        // Convert exponent from 32-bit to 16-bit format
        int32_t new_exponent = int32_t(exponent) - 127 + 15;

        if (new_exponent >= 31) {
            // Overflow to infinity
            return (sign << 15) | 0x7C00;
        }

        if (new_exponent <= 0) {
            // Underflow to zero or subnormal
            if (new_exponent < -10) {
                return sign << 15;
            }
            // Subnormal number
            uint32_t shift = 14 - new_exponent;
            uint32_t subnormal_mantissa = mantissa >> shift;
            return (sign << 15) | subnormal_mantissa;
        }

        // Round mantissa to 10 bits for half precision
        uint32_t rounded_mantissa = (mantissa + 0x1000) >> 13; // Round to nearest

        return (sign << 15) | (new_exponent << 10) | rounded_mantissa;
    }

    static float float16_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                return sign ? -0.0f : 0.0f;
            } else {
                // Subnormal number
                float result = 0.0f;
                uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 13);
                return *reinterpret_cast<float*>(&bits);
            }
        }

        if (exponent == 31) {
            if (mantissa == 0) {
                // Infinity
                uint32_t bits = (sign << 31) | (0xFF << 23);
                return *reinterpret_cast<float*>(&bits);
            } else {
                // NaN
                uint32_t bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                return *reinterpret_cast<float*>(&bits);
            }
        }

        // Normal number
        uint32_t new_exponent = uint32_t(int32_t(exponent) - 15 + 127);
        uint32_t bits = (sign << 31) | (new_exponent << 23) | (mantissa << 13);

        return *reinterpret_cast<float*>(&bits);
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
// it's not an auto-grad framework, but it has backward of the functions
namespace evk::ai {
    struct Pipelines {
        evk::Pipeline matmul_coop;
        evk::Pipeline flash_attn;
        evk::Pipeline flash_attn_bwd;
    };
    std::unique_ptr<Pipelines> pipelines;

    struct MatMulConfig{
        uint16_t k;
        uint16_t n;
        uint8_t transpose_a;
        uint8_t sum_c;
        uint8_t tile_m;
        uint8_t tile_n;
    
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
            .constants = evk::Constant{uint32_t(config.k), uint32_t(config.n), uint32_t(config.transpose_a), uint32_t(config.sum_c), uint32_t(config.tile_m), uint32_t(config.tile_n)},
        });
        // printf("Created matmul pipeline for config: k=%d, n=%d, transpose_a=%d, sum_c=%d, tile_m=%d, tile_n=%d\n", config.k, config.n, config.transpose_a, config.sum_c, config.tile_m, config.tile_n);
        matmul_configs[key] = pipeline;
        return pipeline;
    }

    void initialize() {
        pipelines = std::make_unique<Pipelines>();
        pipelines->matmul_coop = evk::CreatePipeline({
            .name = "matmul_coop",
            .CS = evk::loadSpirvFile("shaders/bin/matmul_coop.comp.spv"),
        });
        pipelines->flash_attn = evk::CreatePipeline({
            .name = "flash_attention",
            .CS = evk::loadSpirvFile("shaders/bin/flash_attention.comp.spv"),
        });
        pipelines->flash_attn_bwd = evk::CreatePipeline({
            .name = "flash_attention_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/flash_attention_bwd.comp.spv"),
        });
    }
    void shutdown() {
        pipelines.reset();
        matmul_configs.clear();
    }

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    void matmul(Tensor& a, Tensor& b, Tensor& c, uint8_t TILE_M = 80u, uint8_t TILE_N = 80u) {
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
        evk::CmdBind(get_matmul_pipeline(MatMulConfig{uint16_t(K), uint16_t(N), 0, 0, TILE_M, TILE_N}));
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
}

struct Graph {
    std::vector<std::unique_ptr<Tensor>> nodes;

    Tensor& tensor(Shape shape, bool param = false) {
        nodes.push_back(std::make_unique<Tensor>(shape));
        Tensor& tensor = *nodes.back();
        return tensor;
    }

    Tensor& matmul(Tensor& a, Tensor& b) {
        nodes.push_back(std::make_unique<Tensor>(Shape({a.shape[0], b.shape[1]})));
        Tensor& tensor = *nodes.back();
        tensor.forward_fn = [this, &a, &b, &tensor]() {
            evk::ai::matmul(a, b, tensor);
        };
        tensor.backward_fn = [this, &a, &b, &tensor]() {
            // TODO: matmul backward
        };
        return tensor;
    }

    void forward() {
        for(auto& node : nodes) {
            if (node->forward_fn) {
                node->forward_fn();
            }
        }
    }
    void backward() {
        for(auto& node : nodes) {
            if (node->backward_fn) {
               node->backward_fn();
            }
        }
    }
};

void test_graph() {
    Graph graph;

    const uint32_t SIZE = 4000u;
    Tensor& a = graph.tensor({SIZE, SIZE});
    {
        float16_t* a_cpu = a.cpu();
        for(uint32_t i = 0; i < a.shape[0]; ++i) {
            a_cpu[i] = float(2);
        }
        a.cpu_upload();
    }
    Tensor& b = graph.tensor({SIZE, SIZE});
    {
        float16_t* b_cpu = b.cpu();
        for(uint32_t i = 0; i < b.shape[0]; ++i) {
            b_cpu[i*b.shape[0]] = float(3);
        }
        b.cpu_upload();
    }
    Tensor& c = graph.matmul(a, b);
    graph.forward();

    a.print(16);
    b.print(16);
    c.print(16);
}

void test_matmul() {
    const uint32_t SIZE = 4096;

    for (uint32_t tile_m = 48; tile_m <= 224; tile_m += 16) {
        for (uint32_t tile_n = 48; tile_n <= 224; tile_n += 16) {
            const uint32_t M = ((SIZE + tile_m - 1) / tile_m) * tile_m;
            const uint32_t N = ((SIZE + tile_n - 1) / tile_n) * tile_n;
            if(tile_m * tile_n >= 128*112) {
                // printf("skipping %d x %d\n", tile_m, tile_n);
                continue;
            }
            Tensor* a = new Tensor({M, SIZE});
            Tensor* b = new Tensor({SIZE, N});
            Tensor* c = new Tensor({M, N});
        
            a->identity(2.0f);
            b->identity(3.0f);
        
            // warm up
            for (uint32_t i = 0; i < 4; ++i) {
                evk::ai::matmul(*a, *b, *c, tile_m, tile_n);
            }

            float min_ms = 1e9f;
            for(int it = 0; it < 1; ++it) {
                const uint32_t subIter = 32;
                for (uint32_t i = 0; i < subIter; ++i) {
                    evk::CmdTimestamp("matmul", [&]() {
                            evk::ai::matmul(*a, *b, *c, tile_m, tile_n);
                    });
                }
                
                evk::Sync();
                for(auto ts: evk::GetTimestamps()) {
                    min_ms = fminf(min_ms, float(ts.end - ts.start)/float(1));
                }
            }
            float tflops = float(2.0f * M * N * M) / (min_ms / 1000.0f) / 1e12f;
            printf("matmul: %5.3fms (%7.3ftflops)", min_ms, tflops);
            printf(" M = %d, N = %d, tile_m = %d, tile_n = %d\n", M, N, tile_m, tile_n);

            delete a;
            delete b;
            delete c;
            evk::Sync();
        }
    }

    // a->print();
    // b->print();
    // c->print();
    // bool c_correct = true;
    // c->cpu_download();
    // float16_t* c_cpu = c->cpu();
    // for(uint32_t i = 0; i < c->shape[0]; ++i) {
    //     int idx = i * (c->shape[0] + 1);
    //     if(c_cpu[idx].value != float16_t(6.0f).value) {
    //         int m = idx / c->shape[0];
    //         int n = idx % c->shape[0];
    //         printf("a * b = c\n");
    //         printf("c[%d, %d] = %f (expected 6.0f)\n", m, n, float(c_cpu[idx]));
    //         c_correct = false;
    //         break;
    //     }
    // }

    // if(c_correct) {
    //     printf("Matrix multiplication is correct!\n");
    // } else {
    //     printf("Failed: Matrix multiplication is not correct!\n");
    // }

    // delete a;
    // delete b;
    // delete c;
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

    test_graph();
    test_matmul();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
