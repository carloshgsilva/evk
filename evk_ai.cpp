#include "evk_ai.h"

namespace evk::ai {
    struct Pipelines {
        evk::Pipeline flash_attn;
        evk::Pipeline flash_attn_bwd;
        evk::Pipeline mse_loss;
        evk::Pipeline cross_entropy;
        evk::Pipeline sgd;
        evk::Pipeline adam;
        evk::Pipeline add;
        evk::Pipeline softmax;
        evk::Pipeline softmax_bwd;
        evk::Pipeline embed;
        evk::Pipeline embed_bwd;
        evk::Pipeline position_add;
        evk::Pipeline position_add_bwd;
        evk::Pipeline causal_mask;
        evk::Pipeline relu;
        evk::Pipeline relu_bwd;
        evk::Pipeline scale;
        evk::Pipeline zero;
        evk::Pipeline sum_batch;
        evk::Pipeline rms_norm;
        evk::Pipeline rms_norm_bwd;
        evk::Buffer flash_scratch;
        uint32_t flash_scratch_elems = 0;
    };
    static std::unique_ptr<Pipelines> pipelines;

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
        uint32_t stride_a;
        uint32_t stride_b;
        uint32_t stride_c;

        operator uint64_t() const {
            return hash_combine(m, k, n, tile_m, tile_n, acc_c, transpose_a, transpose_b, stride_a, stride_b, stride_c);
        }
    };
    
    static std::unordered_map<uint64_t, evk::Pipeline> matmul_configs;
    
    static evk::Pipeline get_matmul_pipeline(MatMulConfig config) {
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
                uint32_t(config.stride_a),
                uint32_t(config.stride_b),
                uint32_t(config.stride_c),
            },
        });
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
    static std::unordered_map<uint64_t, evk::Pipeline> flash_configs;

    static evk::Pipeline get_flash_pipeline(const FlashConfig& config) {
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

    void AdamState::init(uint32_t num_elements) {
        uint32_t size = num_elements * sizeof(float16_t);
        m_buffer = evk::CreateBuffer({
            .size = size,
            .usage = evk::BufferUsage::Storage | evk::BufferUsage::TransferDst,
        });
        v_buffer = evk::CreateBuffer({
            .size = size,
            .usage = evk::BufferUsage::Storage | evk::BufferUsage::TransferDst,
        });
        // Zero-initialize moment buffers
        evk::Buffer zero_buf = evk::CreateBuffer({
            .size = size,
            .usage = evk::BufferUsage::TransferSrc,
            .memoryType = evk::MemoryType::CPU,
        });
        memset(zero_buf.GetPtr(), 0, size);
        evk::CmdCopy(zero_buf, m_buffer, size);
        evk::CmdCopy(zero_buf, v_buffer, size);
        evk::Sync();
        t = 0;
    }

    void AdamState::reset() {
        m_buffer = {};
        v_buffer = {};
        t = 0;
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
        pipelines->adam = evk::CreatePipeline({
            .name = "adam",
            .CS = evk::loadSpirvFile("shaders/bin/adam.comp.spv"),
        });
        pipelines->add = evk::CreatePipeline({
            .name = "add",
            .CS = evk::loadSpirvFile("shaders/bin/add.comp.spv"),
        });
        pipelines->softmax = evk::CreatePipeline({
            .name = "softmax",
            .CS = evk::loadSpirvFile("shaders/bin/softmax.comp.spv"),
        });
        pipelines->softmax_bwd = evk::CreatePipeline({
            .name = "softmax_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/softmax_bwd.comp.spv"),
        });
        pipelines->cross_entropy = evk::CreatePipeline({
            .name = "cross_entropy",
            .CS = evk::loadSpirvFile("shaders/bin/cross_entropy.comp.spv"),
        });
        pipelines->embed = evk::CreatePipeline({
            .name = "embed",
            .CS = evk::loadSpirvFile("shaders/bin/embed.comp.spv"),
        });
        pipelines->embed_bwd = evk::CreatePipeline({
            .name = "embed_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/embed_bwd.comp.spv"),
        });
        pipelines->position_add = evk::CreatePipeline({
            .name = "position_add",
            .CS = evk::loadSpirvFile("shaders/bin/position_add.comp.spv"),
        });
        pipelines->position_add_bwd = evk::CreatePipeline({
            .name = "position_add_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/position_add_bwd.comp.spv"),
        });
        pipelines->causal_mask = evk::CreatePipeline({
            .name = "causal_mask",
            .CS = evk::loadSpirvFile("shaders/bin/causal_mask.comp.spv"),
        });
        pipelines->relu = evk::CreatePipeline({
            .name = "relu",
            .CS = evk::loadSpirvFile("shaders/bin/relu.comp.spv"),
        });
        pipelines->relu_bwd = evk::CreatePipeline({
            .name = "relu_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/relu_bwd.comp.spv"),
        });
        pipelines->scale = evk::CreatePipeline({
            .name = "scale",
            .CS = evk::loadSpirvFile("shaders/bin/scale.comp.spv"),
        });
        pipelines->zero = evk::CreatePipeline({
            .name = "zero",
            .CS = evk::loadSpirvFile("shaders/bin/zero.comp.spv"),
        });
        pipelines->sum_batch = evk::CreatePipeline({
            .name = "sum_batch",
            .CS = evk::loadSpirvFile("shaders/bin/sum_batch.comp.spv"),
        });
        pipelines->rms_norm = evk::CreatePipeline({
            .name = "rms_norm",
            .CS = evk::loadSpirvFile("shaders/bin/rms_norm.comp.spv"),
        });
        pipelines->rms_norm_bwd = evk::CreatePipeline({
            .name = "rms_norm_bwd",
            .CS = evk::loadSpirvFile("shaders/bin/rms_norm_bwd.comp.spv"),
        });
        pipelines->flash_scratch = {};
        pipelines->flash_scratch_elems = 0;
    }

    void shutdown() {
        pipelines.reset();
        matmul_configs.clear();
        flash_configs.clear();
    }

    void matmul(Tensor& a, Tensor& b, Tensor& c, bool transpose_a, bool transpose_b, bool acc_c, uint8_t TILE_M, uint8_t TILE_N) {
        const uint32_t TILE_K = 16u;

        // Basic rank and batch compatibility
        assert(a.shape.rank() >= 2);
        assert(b.shape.rank() >= 2);
        assert(c.shape.rank() >= 2);
        uint32_t batchA = (a.shape.rank() >= 3) ? a.shape.batch_size(2) : 1u;
        uint32_t batchB = (b.shape.rank() >= 3) ? b.shape.batch_size(2) : 1u;
        uint32_t batchC = (c.shape.rank() >= 3) ? c.shape.batch_size(2) : 1u;
        uint32_t batch = (std::max)({batchA, batchB, batchC});
        // Allow broadcasting for inputs; C must match final batch if batch>1
        assert(batchA == batch || batchA == 1u);
        assert(batchB == batch || batchB == 1u);
        assert(batchC == batch || (batch == 1u && batchC == 1u));

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

        // Compute per-batch strides in elements and allow broadcast via zero stride for inputs
        uint32_t elemsA = a.shape[-2] * a.shape[-1];
        uint32_t elemsB = b.shape[-2] * b.shape[-1];
        uint32_t elemsC = c.shape[-2] * c.shape[-1];
        uint32_t strideA = (batchA == 1u && batch > 1u) ? 0u : elemsA;
        uint32_t strideB = (batchB == 1u && batch > 1u) ? 0u : elemsB;
        uint32_t strideC = elemsC; // output cannot be broadcast across batches

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
            .stride_a = strideA,
            .stride_b = strideB,
            .stride_c = strideC,
        }));
        evk::CmdDispatch(tilesCols, tilesRows, batch);
        evk::CmdBarrier();
    }

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

    void flash_attention_bwd(Tensor& q, Tensor& k, Tensor& v, Tensor& o, Tensor& dO, Tensor& dQ, Tensor& dK, Tensor& dV, uint32_t heads) {
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

    void adam(Tensor& param, Tensor& gradient, AdamState& state,
              float learning_rate,
              float beta1,
              float beta2,
              float epsilon) {
        assert(param.shape.rank() == gradient.shape.rank());
        for (uint32_t i = 0; i < param.shape.rank(); ++i) {
            assert(param.shape[i] == gradient.shape[i]);
        }

        uint32_t totalElements = param.shape.count();

        // Initialize state if needed
        if (!state.m_buffer) {
            state.init(totalElements);
        }

        // Increment timestep
        state.t++;

        // Compute bias correction terms
        float beta1_t = std::pow(beta1, float(state.t));
        float beta2_t = std::pow(beta2, float(state.t));
        float beta1CorrectionInv = 1.0f / (1.0f - beta1_t);
        float beta2CorrectionInv = 1.0f / (1.0f - beta2_t);

        evk::CmdBind(pipelines->adam);
        evk::CmdPush(evk::Constant{
            param.buffer.GetReference(),
            gradient.buffer.GetReference(),
            state.m_buffer.GetReference(),
            state.v_buffer.GetReference(),
            learning_rate,
            beta1,
            beta2,
            beta1CorrectionInv,
            beta2CorrectionInv,
            epsilon,
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
    }

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

    void softmax_backward(Tensor& probs, Tensor& grad_out, Tensor& grad_in, float scale_factor) {
        assert(probs.shape.rank() == grad_out.shape.rank());
        assert(grad_in.shape.rank() == grad_out.shape.rank());

        uint32_t lastDim = probs.shape[-1];
        uint32_t outerCount = probs.shape.count() / lastDim;

        evk::CmdBind(pipelines->softmax_bwd);
        evk::CmdPush(evk::Constant{
            probs.buffer.GetReference(),
            grad_out.buffer.GetReference(),
            grad_in.buffer.GetReference(),
            scale_factor,
            lastDim,
            outerCount,
        });
        evk::CmdDispatch(outerCount, 1, 1);
        evk::CmdBarrier();
    }

    void relu(Tensor& in, Tensor& out) {
        assert(in.shape.rank() == out.shape.rank());
        for (uint32_t i = 0; i < in.shape.rank(); ++i) {
            assert(in.shape[i] == out.shape[i]);
        }

        uint32_t totalElements = in.shape.count();

        evk::CmdBind(pipelines->relu);
        evk::CmdPush(evk::Constant{
            in.buffer.GetReference(),
            out.buffer.GetReference(),
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    void relu_backward(Tensor& grad_out, Tensor& in, Tensor& grad_in) {
        assert(grad_out.shape.rank() == in.shape.rank());
        assert(grad_in.shape.rank() == in.shape.rank());

        uint32_t totalElements = in.shape.count();

        evk::CmdBind(pipelines->relu_bwd);
        evk::CmdPush(evk::Constant{
            grad_out.buffer.GetReference(),
            in.buffer.GetReference(),
            grad_in.buffer.GetReference(),
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    void cross_entropy_loss(Tensor& logits, Tensor& targets, Tensor& grad, Tensor& result) {
        assert(logits.shape.rank() == 2);
        assert(targets.shape.rank() == 1);
        assert(result.shape.rank() == 1 && result.shape[0] == 1);
        
        uint32_t totalPositions = logits.shape[0];
        uint32_t vocabSize = logits.shape[1];
        assert(targets.shape[0] == totalPositions);
        assert(grad.shape[0] == totalPositions && grad.shape[1] == vocabSize);

        // Zero out the result buffer first
        float16_t* rp = result.cpu();
        rp[0] = float16_t(0.0f);
        result.cpu_upload();

        evk::CmdBind(pipelines->cross_entropy);
        evk::CmdPush(evk::Constant{
            logits.buffer.GetReference(),
            targets.buffer.GetReference(),
            result.buffer.GetReference(),
            grad.buffer.GetReference(),
            vocabSize,
            totalPositions,
        });
        evk::CmdDispatch(1, 1, 1);
        evk::CmdBarrier();
    }

    void embed(Tensor& embeddings, Tensor& indices, Tensor& out) {
        assert(embeddings.shape.rank() == 2);
        uint32_t vocab_size = embeddings.shape[0];
        uint32_t embed_dim = embeddings.shape[1];
        
        uint32_t num_indices = indices.shape.count();
        assert(out.shape.count() == num_indices * embed_dim);

        uint32_t totalElements = num_indices * embed_dim;

        evk::CmdBind(pipelines->embed);
        evk::CmdPush(evk::Constant{
            embeddings.buffer.GetReference(),
            indices.buffer.GetReference(),
            out.buffer.GetReference(),
            embed_dim,
            num_indices,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    void embed_backward(Tensor& grad_out, Tensor& indices, Tensor& grad_embeddings) {
        assert(grad_embeddings.shape.rank() == 2);
        uint32_t vocab_size = grad_embeddings.shape[0];
        uint32_t embed_dim = grad_embeddings.shape[1];
        
        uint32_t num_indices = indices.shape.count();
        assert(grad_out.shape.count() == num_indices * embed_dim);

        evk::CmdBind(pipelines->embed_bwd);
        evk::CmdPush(evk::Constant{
            grad_out.buffer.GetReference(),
            indices.buffer.GetReference(),
            grad_embeddings.buffer.GetReference(),
            embed_dim,
            num_indices,
        });

        // Dispatch one workgroup per embedding dimension (sequential over indices to avoid races)
        evk::CmdDispatch(embed_dim, 1, 1);
        evk::CmdBarrier();
    }

    void apply_causal_mask(Tensor& scores) {
        assert(scores.shape.rank() >= 2);
        uint32_t N = scores.shape[-1];
        assert(scores.shape[-2] == N);
        
        uint32_t batch = scores.shape.count() / (N * N);
        uint32_t totalElements = batch * N * N;

        evk::CmdBind(pipelines->causal_mask);
        evk::CmdPush(evk::Constant{
            scores.buffer.GetReference(),
            batch,
            N,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // Position embedding addition: out = input + pos_emb (broadcast across batch)
    void position_add(Tensor& input, Tensor& pos_emb, Tensor& out,
                      uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim) {
        uint32_t totalElements = batch_size * seq_len * embed_dim;

        evk::CmdBind(pipelines->position_add);
        evk::CmdPush(evk::Constant{
            input.buffer.GetReference(),
            pos_emb.buffer.GetReference(),
            out.buffer.GetReference(),
            batch_size,
            seq_len,
            embed_dim,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // Position embedding backward
    void position_add_backward(Tensor& grad_out, Tensor& grad_input, Tensor& grad_pos,
                               uint32_t batch_size, uint32_t seq_len, uint32_t embed_dim) {
        const uint32_t WORKGROUP_SIZE = 256u;
        
        // Mode 0: grad_input accumulation
        uint32_t totalElements = batch_size * seq_len * embed_dim;
        evk::CmdBind(pipelines->position_add_bwd);
        evk::CmdPush(evk::Constant{
            grad_out.buffer.GetReference(),
            grad_input.buffer.GetReference(),
            grad_pos.buffer.GetReference(),
            batch_size,
            seq_len,
            embed_dim,
            0u, // mode = 0
        });
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();

        // Mode 1: grad_pos accumulation  
        uint32_t posElements = seq_len * embed_dim;
        evk::CmdBind(pipelines->position_add_bwd);
        evk::CmdPush(evk::Constant{
            grad_out.buffer.GetReference(),
            grad_input.buffer.GetReference(),
            grad_pos.buffer.GetReference(),
            batch_size,
            seq_len,
            embed_dim,
            1u, // mode = 1
        });
        groupsX = (posElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // In-place scale: buffer *= scale
    void scale(Tensor& tensor, float scale_factor) {
        uint32_t totalElements = tensor.shape.count();

        evk::CmdBind(pipelines->scale);
        evk::CmdPush(evk::Constant{
            tensor.buffer.GetReference(),
            scale_factor,
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // Zero out a tensor on GPU
    void zero(Tensor& tensor) {
        uint32_t totalElements = tensor.shape.count();

        evk::CmdBind(pipelines->zero);
        evk::CmdPush(evk::Constant{
            tensor.buffer.GetReference(),
            totalElements,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (totalElements + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // Sum across batch dimension: out[i] += sum_b(input[b, i])
    void sum_batch(Tensor& input, Tensor& output, uint32_t batch_count, uint32_t size_per_batch) {
        evk::CmdBind(pipelines->sum_batch);
        evk::CmdPush(evk::Constant{
            input.buffer.GetReference(),
            output.buffer.GetReference(),
            batch_count,
            size_per_batch,
        });

        const uint32_t WORKGROUP_SIZE = 256u;
        uint32_t groupsX = (size_per_batch + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        evk::CmdDispatch(groupsX, 1, 1);
        evk::CmdBarrier();
    }

    // RMS Normalization forward: out = input / sqrt(mean(input^2) + eps)
    // input: (*, D) where * is any batch dimensions, D is the last dimension to normalize
    void rms_norm(Tensor& input, Tensor& output, float eps) {
        assert(input.shape.rank() == output.shape.rank());
        for (uint32_t i = 0; i < input.shape.rank(); ++i) {
            assert(input.shape[i] == output.shape[i]);
        }

        uint32_t D = input.shape[-1];
        uint32_t outerCount = input.shape.count() / D;

        evk::CmdBind(pipelines->rms_norm);
        evk::CmdPush(evk::Constant{
            input.buffer.GetReference(),
            output.buffer.GetReference(),
            eps,
            D,
            outerCount,
        });

        // One workgroup per row
        evk::CmdDispatch(outerCount, 1, 1);
        evk::CmdBarrier();
    }

    // RMS Normalization backward
    // Gradient: dL/dx_i = (g_i - y_i * dot(g,y) / D) / rms
    // Accumulates into grad_input
    void rms_norm_backward(Tensor& input, Tensor& grad_out, Tensor& grad_input, float eps) {
        assert(input.shape.rank() == grad_out.shape.rank());
        assert(input.shape.rank() == grad_input.shape.rank());

        uint32_t D = input.shape[-1];
        uint32_t outerCount = input.shape.count() / D;

        evk::CmdBind(pipelines->rms_norm_bwd);
        evk::CmdPush(evk::Constant{
            input.buffer.GetReference(),
            grad_out.buffer.GetReference(),
            grad_input.buffer.GetReference(),
            eps,
            D,
            outerCount,
        });

        // One workgroup per row
        evk::CmdDispatch(outerCount, 1, 1);
        evk::CmdBarrier();
    }
}

