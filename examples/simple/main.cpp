#include <iostream>
#include <evk.h>
#include <cassert>

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

    void print() {
        printf("Tensor (");
        for (uint32_t i = 0; i < shape.rank(); ++i) {
            if(i != 0) printf(", ");
            printf("%d", shape[i]);
        }
        printf("): [");
        std::vector<float>& data = cpu();
        for (uint32_t i = 0; i < shape.count(); ++i) {
            if(i != 0) printf(" ");
            printf("%f", data[i]);
        }
        printf("]\n");
    }
};

// pure fp16 and u16 tensors machine learning library
// it's not an auto-grad framework, but it has backward of the functions
namespace evk::ai {
    struct Pipelines {
        evk::Pipeline matmul;
        evk::Pipeline matmul_bwd;
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

        evk::CmdBind(pipelines->matmul);
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

        // dispatch groups based on TILE size in shader (TILE = 16)
        const uint32_t TILE = 16u;
        uint32_t groupX = (N + TILE - 1) / TILE; // covers columns (N)
        uint32_t groupY = (M + TILE - 1) / TILE; // covers rows (M)
        evk::CmdDispatch(groupX, groupY, batch);
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
        evk::CmdPush(evk::Constant{
            a_transposed.buffer.GetRID(),
            c_grad.buffer.GetRID(),
            b_grad.buffer.GetRID()
        });
        // Dispatch sizes must match forward's layout. Recompute dims from buffers' shapes.
        uint32_t batch = 1;
        if (a_transposed.shape.rank() >= 3) batch = a_transposed.shape.batch_size(2);

        uint32_t M = (a_transposed.shape.rank() >= 1) ? a_transposed.shape[0] : 1; // rows
        uint32_t K = (a_transposed.shape.rank() >= 2) ? a_transposed.shape[1] : 1; // cols
        uint32_t N = (c_grad.shape.rank() >= 2) ? c_grad.shape[1] : 1;

        const uint32_t TILE = 16u;
        uint32_t groupX = (N + TILE - 1) / TILE;
        uint32_t groupY = (M + TILE - 1) / TILE;
        evk::CmdDispatch(groupX, groupY, batch);
    }
}


struct Layer {
    // MLP
    // Attention

    std::unique_ptr<Tensor> one;
};

struct Transformer {
    std::unique_ptr<Tensor> input;
    std::unique_ptr<Tensor> weights;
    // std::vector<Layer> layers;
    std::unique_ptr<Tensor> output;
    std::unique_ptr<Tensor> output_grad;
    // std::unique_ptr<Tensor> weights_grad;

    Transformer(uint32_t num_layers) {
        // for (uint32_t i = 0; i < num_layers; ++i) {
        //     layers.push_back(Layer());
        // }
        weights = std::make_unique<Tensor>(Shape({2, 2}));
        input = std::make_unique<Tensor>(Shape({2, 2}));
        output = std::make_unique<Tensor>(Shape({2, 2}));
        output_grad = std::make_unique<Tensor>(Shape({2, 2}));
        // weights_grad = std::make_unique<Tensor>(Shape({2, 2}));
    }

    void forward() {
        evk::ai::matmul(*input, *weights, *output);
        // for (auto& layer : layers) {
        // }
        evk::Sync();
    }

    void backward() {
        std::vector<float> target_output = {-2.0f, -4.0f, -6.0f, -8.0f};
        std::vector<float>& output_cpu = output->cpu();
        std::vector<float>& output_grad_cpu = output_grad->cpu();
        for (uint32_t i = 0; i < target_output.size(); ++i) {
            output_grad_cpu[i] = (target_output[i] - output_cpu[i])*0.05f;
        }
        output_grad->set_cpu(output_grad_cpu);

        evk::ai::matmul_bwd(*input, *output_grad, *weights);
        // for (auto& layer : layers) {
        // }
        evk::Sync();
    }
};


void test_graph() {
    evk::ai::initialize();

    Transformer transformer(1);
    transformer.input->set_cpu({1.0f, 2.0f, 3.0f, 4.0f});
    transformer.weights->set_cpu({1.0f, 2.0f, 3.0f, 4.0f});

    transformer.input->print();
    transformer.weights->print();
    
    for (uint32_t i = 0; i < 2500; ++i) {
        printf("[iteration %d]\n", i);
        transformer.forward();
        transformer.backward();
        printf("Weights: ");
        transformer.weights->print();
        printf("Output: ");
        transformer.output->print();
        printf("Output grad: ");
        transformer.output_grad->print();
    }
    
    // printf("input.grad[0] = %f\n", transformer.input->grad[0]);
    // printf("weights.grad[0] = %f\n", transformer.weights->grad[0]);
    // printf("output.grad[0] = %f\n", transformer.output->grad[0]);
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

    test_graph();

    evk::ai::shutdown();
    evk::Shutdown();
    return 0;
}
