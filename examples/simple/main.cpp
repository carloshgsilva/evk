#include <iostream>
#include <evk.h>

#include "win_dbg.h"

struct Shape {
    uint64_t values[8];
    uint32_t count;

    uint64_t operator[] (uint64_t index) const {
        return values[index];
    }
};

struct Tensor {
    float* buffer;
    uint64_t size;
    Shape shape;
    bool is_param;
    float* grad;

    void optimizer_zero() {
        if (grad) {
            for (uint64_t i = 0; i < size; ++i) grad[i] = 0.0f;
        }
    }
    void optimizer_step() {
        if (is_param && grad) {
            // simple SGD with lr = 0.01
            const float lr = 0.01f;
            for (uint64_t i = 0; i < size; ++i) buffer[i] -= lr * grad[i];
        }
    }

    Tensor(const Shape& shape, bool is_param = false) {
        this->shape = shape;
        this->is_param = is_param;
        // compute total size as product of first `count` dimensions
        uint64_t s = 1;
        for (uint32_t i = 0; i < shape.count; ++i) s *= shape.values[i];
        this->size = s;
        buffer = new float[s];
        grad = new float[s];
        // initialize to zero
        for (uint64_t i = 0; i < s; ++i) { buffer[i] = 0.0f; grad[i] = 0.0f; }
    }

    virtual void forward() {
        // base tensor has no-op forward
    }
    virtual void backward() {
        // base tensor has no-op backward
    }
};

struct MatMul : Tensor {
    Tensor* a;
    Tensor* b;
    MatMul(Tensor* a, Tensor* b) : Tensor(Shape{}, false) {
        // set shape properly
        Shape s;
        s.count = 2;
        s.values[0] = a->shape.values[0];
        s.values[1] = b->shape.values[1];
        this->shape = s;
        uint64_t sz = s.values[0] * s.values[1];
        this->size = sz;
        delete[] buffer;
        delete[] grad;
        buffer = new float[sz];
        grad = new float[sz];
        for (uint64_t i = 0; i < sz; ++i) { buffer[i] = 0.0f; grad[i] = 0.0f; }
        this->a = a;
        this->b = b;
    }

    void forward() {
        // assume a: (m x k), b: (k x n) -> this: (m x n)
        uint64_t m = a->shape.values[0];
        uint64_t k = a->shape.values[1];
        uint64_t n = b->shape.values[1];
        for (uint64_t i = 0; i < m; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (uint64_t t = 0; t < k; ++t) {
                    sum += a->buffer[i*k + t] * b->buffer[t*n + j];
                }
                buffer[i*n + j] = sum;
            }
        }
    }

    void backward() {
        printf("MatMul backward\n");
        // compute gradients w.r.t a and b assuming grad contains dL/dC
        uint64_t m = a->shape.values[0];
        uint64_t k = a->shape.values[1];
        uint64_t n = b->shape.values[1];
        // accumulate into a->grad and b->grad
        for (uint64_t i = 0; i < m; ++i) {
            for (uint64_t t = 0; t < k; ++t) {
                float acc = 0.0f;
                for (uint64_t j = 0; j < n; ++j) {
                    acc += grad[i*n + j] * b->buffer[t*n + j];
                }
                a->grad[i*k + t] += acc;
            }
        }
        for (uint64_t t = 0; t < k; ++t) {
            for (uint64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (uint64_t i = 0; i < m; ++i) {
                    acc += a->buffer[i*k + t] * grad[i*n + j];
                }
                b->grad[t*n + j] += acc;
            }
        }
    }
};

struct Graph {
    std::vector<Tensor*> tensors;
    evk::Pipeline pipeline = evk::CreatePipeline({
        .name = "MatMul",
        .CS = evk::loadSpirvFile("shaders/bin/matmul.comp.spv"),
    });

    Graph() {
    }
    ~Graph() {
        for (auto t : tensors) {
            delete t;
        }
    }

    Tensor* zero() {
        Shape s; s.count = 2; s.values[0] = 1; s.values[1] = 1;
        Tensor* t = new Tensor(s, false);
        t->buffer[0] = 0.0f;
        tensors.push_back(t);
        return t;
    }
    Tensor* one() {
        Shape s; s.count = 2; s.values[0] = 1; s.values[1] = 1;
        Tensor* t = new Tensor(s, false);
        t->buffer[0] = 1.0f;
        tensors.push_back(t);
        return t;
    }
    Tensor* matmul(Tensor* a, Tensor* b) {
        MatMul* m = new MatMul(a, b);
        tensors.push_back(m); 
        return m;
    }


    void forward() {
        for (auto t : tensors) {
            t->forward();
        }
    }
    void backward() {
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            (*it)->backward();
        }
    }

    // zero the gradients
    void optimizer_zero() {
        for (auto t : tensors) {
            t->optimizer_zero();
        }
    }
    // step the optimizer and apply the gradients
    void optimizer_step() {
        for (auto t : tensors) {
            t->optimizer_step();
        }
    }
};

void test_graph() {
    Graph g;
    Tensor* a = g.one();
    Tensor* b = g.one();
    Tensor* c = g.matmul(a, b);
    g.forward();
    g.optimizer_zero();
    c->grad[0] = 1.0f;                                                                  
    g.backward();
    g.optimizer_step();
 
    printf("a.grad[0] = %f\n", a->grad[0]);
    printf("b.grad[0] = %f\n", b->grad[0]);
    printf("c.grad[0] = %f\n", c->grad[0]);
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

    evk::Shutdown();
    return 0;
}
