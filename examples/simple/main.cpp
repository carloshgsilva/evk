#include <iostream>
#include <evk.h>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "win_dbg.h"

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
    Shape shape;
    std::vector<float> cpu_data;

    Tensor(const Shape& shape) {
        this->shape = shape;
        // compute total size as product of first `count` dimensions
        uint32_t s = shape.count() * sizeof(float);
        buffer = evk::CreateBuffer({
            .size = s,
            .usage = evk::BufferUsage::Storage,
        });
    }

    // copies data from CPU to GPU
    void set_cpu(const std::vector<float>& data) {
        assert(data.size() == shape.count());
        cpu_data = data;
        evk::CmdCopy(cpu_data.data(), buffer, shape.count() * sizeof(float));
        evk::Sync();
    }

    // allocates and copies data from GPU to CPU
    std::vector<float>& cpu() {
        cpu_data.resize(shape.count());
        evk::Buffer cpu_buffer = evk::CreateBuffer({
            .size = shape.count() * sizeof(float),
            .usage = evk::BufferUsage::TransferDst,
            .memoryType = evk::MemoryType::CPU,
        });
        evk::CmdCopy(buffer, cpu_buffer, shape.count() * sizeof(float));
        evk::Sync();
        evk::ReadBuffer(cpu_buffer, cpu_data.data(), shape.count() * sizeof(float), 0);
        return cpu_data;
    }

    void identity(float val = 1.0f) {
        cpu_data.resize(shape.count());
        for (uint32_t i = 0; i < shape.count(); ++i) {
            cpu_data[i] = float((i % (shape[0]+1) == 0)? val : 0.0f);
        }
        set_cpu(cpu_data);
    }

    void print(uint32_t max_elements = 8, uint32_t max_batch = 4) {
        // Print shape header
        printf("Tensor (");
        for (uint32_t i = 0; i < shape.rank(); ++i) {
            if(i != 0) printf(", ");
            printf("%d", shape[i]);
        }
        printf("):\n");

        std::vector<float>& data = cpu();

        // If rank < 2 just fallback to flat print (limited)
        if (shape.rank() < 2) {
            uint32_t to_show = (std::min)(shape.count(), max_elements);
            printf("[");
            for (uint32_t i = 0; i < to_show; ++i) {
                if (i) printf(", ");
                printf("%g", data[i]);
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
                    printf("%g", data[row_offset + c]);
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
                    printf("%g", data[last_row_offset + c]);
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
};

// pure fp16 and u16 tensors machine learning library
// it's not an auto-grad framework, but it has backward of the functions
namespace evk::ai {
    struct Pipelines {
        evk::Pipeline matmul;
        evk::Pipeline matmul_bwd;
        evk::Pipeline matmul_coop;
        evk::Pipeline flash_attn;
        evk::Pipeline flash_attn_bwd;
    };
    std::unique_ptr<Pipelines> pipelines;

    void initialize() {
        pipelines = std::make_unique<Pipelines>();
        pipelines->matmul = evk::CreatePipeline({
            .name = "matmul",
            .CS = evk::loadSpirvFile("shaders/bin/matmul.comp.spv"),
        });
        pipelines->matmul_bwd = evk::CreatePipeline({
            .name = "matmul_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/matmul.comp.spv"),
            .constants = evk::Constant{1, 1}, // TRANSPOSE_A = 1, SUM_C = 1
        });
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
    }

    // C = A * B
    // (...B, M, N) = (...B, M, K) * (...B, K, N)
    void matmul(Tensor& a, Tensor& b, Tensor& c) {
        assert(a.shape.rank() >= 2);
        assert(b.shape.rank() >= 2);
        assert(c.shape.rank() >= 2);
        assert(a.shape.batch_size(2) == b.shape.batch_size(2));
        assert(a.shape[-1] == b.shape[-2]);

        // Determine batch size B. If tensor rank >= 3 use first dim as batch, otherwise 1.
        uint32_t batch = 1;
        if (a.shape.rank() >= 3) batch = a.shape.batch_size(2);

        // infer matrix dims: A is M x K, B is K x N
        uint32_t M = (a.shape.rank() >= 1) ? a.shape[0] : 1;
        uint32_t K = (a.shape.rank() >= 2) ? a.shape[1] : 1;
        uint32_t N = (b.shape.rank() >= 2) ? b.shape[1] : 1;

        evk::CmdPush(evk::Constant{
            a.buffer.GetRID(),
            b.buffer.GetRID(),
            c.buffer.GetRID(),
            batch,
            M,
            K,
            N
        });

        constexpr bool USE_COOP = true;
        if (USE_COOP) {
            const uint32_t TILE = 96u;
            uint32_t tilesCols = (N + TILE - 1) / TILE; // columns (N)
            uint32_t tilesRows = (M + TILE - 1) / TILE; // rows (M)
            // Map: X=tilesCols, Y=tilesRows, Z=batch
            evk::CmdBind(pipelines->matmul_coop);
            evk::CmdDispatch(tilesCols, tilesRows, batch);
        } else {
            const uint32_t TILE = 16u; // matches matmul.comp local size
            uint32_t tilesCols = (N + TILE - 1) / TILE; // columns (N)
            uint32_t tilesRows = (M + TILE - 1) / TILE; // rows (M)
            // Map: X=tilesCols, Y=tilesRows, Z=batch
            evk::CmdBind(pipelines->matmul);
            evk::CmdDispatch(tilesCols, tilesRows, batch);
        }
    }

    // dL/dB = A^T * dL/dC
    // (...B, M, N) = (...B, K, M)^T * (...B, K, N)
    // a is already transposed
    void matmul_bwd(Tensor& a_transposed, Tensor& c_grad, Tensor& b_grad) {
        assert(a_transposed.shape.rank() >= 2);
        assert(c_grad.shape.rank() >= 2);
        assert(b_grad.shape.rank() >= 2);
        assert(a_transposed.shape.batch_size(2) == c_grad.shape.batch_size(2));
        assert(a_transposed.shape[-2] == c_grad.shape[-2]);

        evk::CmdBind(pipelines->matmul_bwd);
        // Recompute dims under dB = A^T (KxM) * dC (MxN) => dB (KxN)
        uint32_t batch = 1;
        if (a_transposed.shape.rank() >= 3) batch = a_transposed.shape.batch_size(2);

        uint32_t M = (a_transposed.shape.rank() >= 1) ? a_transposed.shape[0] : 1; // K (rows of output)
        uint32_t K = (a_transposed.shape.rank() >= 2) ? a_transposed.shape[1] : 1; // M (shared dim)
        uint32_t N = (c_grad.shape.rank() >= 2) ? c_grad.shape[1] : 1;

        evk::CmdPush(evk::Constant{
            a_transposed.buffer.GetRID(),
            c_grad.buffer.GetRID(),
            b_grad.buffer.GetRID(),
            batch,
            M,
            K,
            N
        });

#if 0
        uint32_t groupX = (N + TILE - 1) / TILE;
        uint32_t groupY = (M + TILE - 1) / TILE;
        evk::CmdDispatch(groupX, groupY, batch);
#else
        const uint32_t TILE = 16u; // matches matmul.comp local size
        uint32_t tilesCols = (N + TILE - 1) / TILE;
        uint32_t tilesRows = (M + TILE - 1) / TILE;
        // X=sizeX, Y=sizeY, Z=batch
        evk::CmdDispatch(tilesCols, tilesRows, batch);
#endif
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


struct Layer {
    // MLP
    // Attention

    // Attn = CatHeads(softmax(Q * K^T) * V) * W0
    std::unique_ptr<Tensor> q;
    std::unique_ptr<Tensor> k;
    std::unique_ptr<Tensor> v;
    std::unique_ptr<Tensor> w0;

    std::unique_ptr<Tensor> mlp_a;
    std::unique_ptr<Tensor> mlp_b;

    std::unique_ptr<Tensor> one;
};

struct Transformer {
    std::unique_ptr<Tensor> input;
    std::unique_ptr<Tensor> weights;
    // std::vector<Layer> layers;
    std::unique_ptr<Tensor> output;
    std::unique_ptr<Tensor> output_grad;
    // std::unique_ptr<Tensor> weights_grad;
    uint32_t test_dim;

    Transformer(uint32_t num_layers, uint32_t test_dim) {
        this->test_dim = test_dim;
        // for (uint32_t i = 0; i < num_layers; ++i) {
        //     layers.push_back(Layer());
        // }
        weights = std::make_unique<Tensor>(Shape({test_dim, test_dim}));
        input = std::make_unique<Tensor>(Shape({test_dim, test_dim}));
        output = std::make_unique<Tensor>(Shape({test_dim, test_dim}));
        output_grad = std::make_unique<Tensor>(Shape({test_dim, test_dim}));
        // weights_grad = std::make_unique<Tensor>(Shape({2, 2}));
    }

    void forward() {
        evk::CmdTimestamp("matmul", [&]() {
            evk::ai::matmul(*input, *weights, *output);
        });

        // for (auto& layer : layers) {
        // }
        evk::Sync();
        for(auto ts: evk::GetTimestamps()) {
            printf("%s:%.3fms ", ts.name, float(ts.end - ts.start));
        }
    }

    void backward() {
        std::vector<float> target_output(test_dim * test_dim);
        for (uint32_t i = 0; i < test_dim * test_dim; ++i) {
            target_output[i] = float(i);
        }
        std::vector<float>& output_cpu = output->cpu();
        std::vector<float>& output_grad_cpu = output_grad->cpu();
        for (uint32_t i = 0; i < target_output.size(); ++i) {
            float target = (i % (test_dim+1) == 0)? 5.0f : 0.0f;
            output_grad_cpu[i] = (target_output[i] - output_cpu[i])*0.005f;
        }
        output_grad->set_cpu(output_grad_cpu);

        evk::ai::matmul_bwd(*input, *output_grad, *weights);
        // for (auto& layer : layers) {
        // }
        evk::Sync();
    }
};


void test_graph() {
    Transformer transformer(1, 16*16*16);
    transformer.input->identity();
    transformer.weights->identity();

    transformer.input->print();
    transformer.weights->print();
    
    for (uint32_t i = 0; i < 5; ++i) {
        printf("[iteration %d] ", i);
        transformer.forward();
        transformer.backward();
        // printf("Weights: ");
        // transformer.weights->print();
        printf("Output: ");
        transformer.output->print();
        // printf("Output grad: ");
        // transformer.output_grad->print();
    }
    
    // printf("input.grad[0] = %f\n", transformer.input->grad[0]);
    // printf("weights.grad[0] = %f\n", transformer.weights->grad[0]);
    // printf("output.grad[0] = %f\n", transformer.output->grad[0]);
}

void test_matmul() {
    const uint32_t size = 16*16*16;
    Tensor* a = new Tensor({size, size});
    Tensor* b = new Tensor({size, size});
    Tensor* c = new Tensor({size, size});

    a->identity(2.0f);
    b->identity(3.0f);

    for (uint32_t i = 0; i < 32; ++i) {
        evk::CmdTimestamp("matmul", [&]() {
            evk::ai::matmul(*a, *b, *c);
        });
    }

    evk::Sync();
    float avg_ms = 0.0f;
    float total_count = 0.0f;
    for(auto ts: evk::GetTimestamps()) {
        avg_ms += float(ts.end - ts.start);
        total_count += 1.0f;
    }
    avg_ms /= total_count;
    float tflops = float(2.0f * size * size * size) / (avg_ms / 1000.0f) / 1e12f;
    printf("matmul: %.3fms (%.3ftflops)\n", avg_ms, tflops);

    // a->print();
    // b->print();
    // c->print();
    if(c->cpu()[0] == 6.0f) {
        printf("Matrix multiplication is correct!\n");
    } else {
        printf("c[0] = %f\n", c->cpu()[0]);
        printf("first c matrix element is not 6.0f, so the matmul is not correct!\n");
        assert(false);
    }

    delete a;
    delete b;
    delete c;
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

    // test_graph();
    test_matmul();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
