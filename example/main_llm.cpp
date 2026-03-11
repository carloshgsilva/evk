#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>

#include <evk_ai.h>

namespace {
constexpr uint32_t kCubeTriangleCount = 12;
constexpr uint32_t kTrianglesPerMesh = 16;
constexpr uint32_t kCoordsPerTriangle = 9;
constexpr uint32_t kMeshFeatureDim = 10; // 9 coords + exist

constexpr uint16_t kPadToken = 0; // also ignore token for CE
constexpr uint16_t kBosToken = 1;
constexpr uint16_t kEosToken = 2;
constexpr uint16_t kCoordTokenBase = 3;
constexpr uint32_t kCoordBins = 128;

constexpr uint32_t kCoordTokenCount = kTrianglesPerMesh * kCoordsPerTriangle; // 144
constexpr uint32_t kSeqActiveLen = 1 + kCoordTokenCount + 1; // BOS + coords + EOS = 146
constexpr uint32_t kSeqLen = 160; // tile-aligned (matmul requires multiples of 16)
constexpr uint32_t kVocabSize = 144; // tile-aligned, uses ids [0..130]

constexpr float kCoordMin = -1.25f;
constexpr float kCoordMax = 1.25f;
constexpr float kExistScale = 1.0f;
constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Mat3 {
    float m[9];
};

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 scale(const Vec3& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

Vec3 mul(const Mat3& r, const Vec3& v) {
    return {
        r.m[0] * v.x + r.m[1] * v.y + r.m[2] * v.z,
        r.m[3] * v.x + r.m[4] * v.y + r.m[5] * v.z,
        r.m[6] * v.x + r.m[7] * v.y + r.m[8] * v.z
    };
}

Mat3 rotation_from_euler(float ax, float ay, float az) {
    float cx = cosf(ax);
    float sx = sinf(ax);
    float cy = cosf(ay);
    float sy = sinf(ay);
    float cz = cosf(az);
    float sz = sinf(az);

    Mat3 r{};
    r.m[0] = cz * cy;
    r.m[1] = cz * sy * sx - sz * cx;
    r.m[2] = cz * sy * cx + sz * sx;

    r.m[3] = sz * cy;
    r.m[4] = sz * sy * sx + cz * cx;
    r.m[5] = sz * sy * cx - cz * sx;

    r.m[6] = -sy;
    r.m[7] = cy * sx;
    r.m[8] = cy * cx;
    return r;
}

float rand_uniform(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

Mat3 random_rotation(std::mt19937& rng) {
    float ax = rand_uniform(rng, 0.0f, 2.0f * kPi);
    float ay = rand_uniform(rng, 0.0f, 2.0f * kPi);
    float az = rand_uniform(rng, 0.0f, 2.0f * kPi);
    return rotation_from_euler(ax, ay, az);
}

std::vector<std::array<float, 9>> make_cube_triangles() {
    Vec3 v0{-0.5f, -0.5f, -0.5f};
    Vec3 v1{ 0.5f, -0.5f, -0.5f};
    Vec3 v2{ 0.5f,  0.5f, -0.5f};
    Vec3 v3{-0.5f,  0.5f, -0.5f};
    Vec3 v4{-0.5f, -0.5f,  0.5f};
    Vec3 v5{ 0.5f, -0.5f,  0.5f};
    Vec3 v6{ 0.5f,  0.5f,  0.5f};
    Vec3 v7{-0.5f,  0.5f,  0.5f};

    auto tri = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        return std::array<float, 9>{a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z};
    };

    return {
        // Bottom (z = -0.5)
        tri(v0, v1, v2),
        tri(v0, v2, v3),
        // Top (z = 0.5)
        tri(v4, v6, v5),
        tri(v4, v7, v6),
        // Front (y = -0.5)
        tri(v0, v5, v1),
        tri(v0, v4, v5),
        // Back (y = 0.5)
        tri(v3, v2, v6),
        tri(v3, v6, v7),
        // Left (x = -0.5)
        tri(v0, v3, v7),
        tri(v0, v7, v4),
        // Right (x = 0.5)
        tri(v1, v5, v6),
        tri(v1, v6, v2)
    };
}

void apply_transform(std::array<float, 9>& tri, const Mat3& r, float s, const Vec3& t) {
    for (int i = 0; i < 3; ++i) {
        Vec3 v{tri[i * 3 + 0], tri[i * 3 + 1], tri[i * 3 + 2]};
        v = mul(r, v);
        v = scale(v, s);
        v = add(v, t);
        tri[i * 3 + 0] = v.x;
        tri[i * 3 + 1] = v.y;
        tri[i * 3 + 2] = v.z;
    }
}

struct MeshBatch {
    std::vector<float> target; // [B, T, 10], 9 coords + exist
};

void generate_batch(const std::vector<std::array<float, 9>>& base,
                    uint32_t batch_size,
                    std::mt19937& rng,
                    MeshBatch& batch) {
    batch.target.assign(batch_size * kTrianglesPerMesh * kMeshFeatureDim, 0.0f);

    std::vector<uint32_t> order(kCubeTriangleCount);

    for (uint32_t b = 0; b < batch_size; ++b) {
        auto tris = base;
        Mat3 rot = random_rotation(rng);
        Vec3 translate{
            rand_uniform(rng, -0.45f, 0.45f),
            rand_uniform(rng, -0.45f, 0.45f),
            rand_uniform(rng, -0.45f, 0.45f)
        };

        for (auto& tri : tris) {
            apply_transform(tri, rot, 1.0f, translate);
        }

        std::iota(order.begin(), order.end(), 0);

        for (uint32_t t = 0; t < kTrianglesPerMesh; ++t) {
            uint32_t tri_idx = order[t % kCubeTriangleCount];
            const auto& tri = tris[tri_idx];
            uint32_t base_idx = (b * kTrianglesPerMesh + t) * kMeshFeatureDim;

            for (uint32_t i = 0; i < 9; ++i) {
                batch.target[base_idx + i] = tri[i];
            }
            batch.target[base_idx + 9] = kExistScale;
        }
    }
}

uint16_t quantize_coord_to_token(float v) {
    float clamped = (std::max)(kCoordMin, (std::min)(kCoordMax, v));
    float norm = (clamped - kCoordMin) / (kCoordMax - kCoordMin);
    float scaled = norm * float(kCoordBins - 1);
    uint32_t bin = uint32_t(std::lround(scaled));
    if (bin >= kCoordBins) {
        bin = kCoordBins - 1;
    }
    return uint16_t(kCoordTokenBase + bin);
}

float dequantize_coord_from_token(uint16_t token) {
    if (token < kCoordTokenBase) {
        return 0.0f;
    }
    uint32_t bin = uint32_t(token - kCoordTokenBase);
    if (bin >= kCoordBins) {
        bin = kCoordBins - 1;
    }
    float norm = float(bin) / float(kCoordBins - 1);
    return kCoordMin + norm * (kCoordMax - kCoordMin);
}

void build_token_batch(const std::vector<float>& mesh_target,
                       uint32_t batch_size,
                       std::vector<uint16_t>& input_tokens,
                       std::vector<uint16_t>& target_tokens) {
    input_tokens.assign(batch_size * kSeqLen, kPadToken);
    target_tokens.assign(batch_size * kSeqLen, kPadToken);

    for (uint32_t b = 0; b < batch_size; ++b) {
        uint16_t* in_ptr = input_tokens.data() + b * kSeqLen;
        uint16_t* tgt_ptr = target_tokens.data() + b * kSeqLen;

        in_ptr[0] = kBosToken;

        for (uint32_t i = 0; i < kCoordTokenCount; ++i) {
            uint32_t tri = i / kCoordsPerTriangle;
            uint32_t coord = i % kCoordsPerTriangle;
            uint32_t idx = (b * kTrianglesPerMesh + tri) * kMeshFeatureDim + coord;
            in_ptr[1 + i] = quantize_coord_to_token(mesh_target[idx]);
        }

        in_ptr[1 + kCoordTokenCount] = kEosToken;

        // Teacher-forcing target: predict next token.
        for (uint32_t p = 0; p <= kCoordTokenCount; ++p) {
            tgt_ptr[p] = in_ptr[p + 1];
        }
    }
}

void upload_token_tensor(Tensor& t, const std::vector<uint16_t>& data) {
    assert(t.shape.count() == data.size());
    float16_t* ptr = t.cpu();
    for (uint32_t i = 0; i < t.shape.count(); ++i) {
        ptr[i].value = data[i];
    }
    t.cpu_upload();
}

void download_float_tensor(Tensor& t, std::vector<float>& data) {
    t.cpu_download();
    float16_t* ptr = t.cpu();
    data.resize(t.shape.count());
    for (uint32_t i = 0; i < t.shape.count(); ++i) {
        data[i] = float(ptr[i]);
    }
}

void decode_generated_mesh(const std::vector<uint16_t>& generated_inputs,
                           uint32_t batch_index,
                           std::vector<float>& mesh_features_out) {
    mesh_features_out.assign(kTrianglesPerMesh * kMeshFeatureDim, 0.0f);

    const uint16_t* in = generated_inputs.data() + batch_index * kSeqLen;
    for (uint32_t i = 0; i < kCoordTokenCount; ++i) {
        uint32_t tri = i / kCoordsPerTriangle;
        uint32_t coord = i % kCoordsPerTriangle;
        uint32_t dst = tri * kMeshFeatureDim + coord;
        mesh_features_out[dst] = dequantize_coord_from_token(in[1 + i]);
    }

    for (uint32_t tri = 0; tri < kTrianglesPerMesh; ++tri) {
        mesh_features_out[tri * kMeshFeatureDim + 9] = kExistScale;
    }
}

void sample_greedy_autoregressive(struct MeshTokenModel& model,
                                  uint32_t batch_size,
                                  std::vector<uint16_t>& generated_inputs,
                                  std::vector<uint16_t>& scratch_targets,
                                  std::vector<float>& logits_cpu);

void export_obj(const std::filesystem::path& path,
                const std::vector<float>& features,
                uint32_t tri_count,
                float exist_threshold = 0.5f) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        return;
    }

    uint32_t vertex_index = 1;
    for (uint32_t t = 0; t < tri_count; ++t) {
        uint32_t base_idx = t * kMeshFeatureDim;
        float exist = features[base_idx + 9] / kExistScale;
        if (exist < exist_threshold) {
            continue;
        }

        out << "v " << features[base_idx + 0] << " " << features[base_idx + 1] << " " << features[base_idx + 2] << "\n";
        out << "v " << features[base_idx + 3] << " " << features[base_idx + 4] << " " << features[base_idx + 5] << "\n";
        out << "v " << features[base_idx + 6] << " " << features[base_idx + 7] << " " << features[base_idx + 8] << "\n";
        out << "f " << vertex_index << " " << vertex_index + 2 << " " << vertex_index + 1 << "\n";
        vertex_index += 3;
    }
}

void append_obj(const std::filesystem::path& path,
                const std::vector<float>& features,
                uint32_t tri_count,
                uint32_t& vertex_index,
                float x_offset = 0.0f,
                float y_offset = 0.0f,
                float exist_threshold = 0.5f) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        return;
    }

    for (uint32_t t = 0; t < tri_count; ++t) {
        uint32_t base_idx = t * kMeshFeatureDim;
        float exist = features[base_idx + 9] / kExistScale;
        if (exist < exist_threshold) {
            continue;
        }

        float x0 = features[base_idx + 0] + x_offset;
        float y0 = features[base_idx + 1] + y_offset;
        float z0 = features[base_idx + 2];
        float x1 = features[base_idx + 3] + x_offset;
        float y1 = features[base_idx + 4] + y_offset;
        float z1 = features[base_idx + 5];
        float x2 = features[base_idx + 6] + x_offset;
        float y2 = features[base_idx + 7] + y_offset;
        float z2 = features[base_idx + 8];

        out << "v " << x0 << " " << y0 << " " << z0 << "\n";
        out << "v " << x1 << " " << y1 << " " << z1 << "\n";
        out << "v " << x2 << " " << y2 << " " << z2 << "\n";
        out << "f " << vertex_index << " " << vertex_index + 2 << " " << vertex_index + 1 << "\n";
        vertex_index += 3;
    }
}

struct CausalAttentionBlock {
    Tensor* w_q = nullptr;
    Tensor* w_k = nullptr;
    Tensor* w_v = nullptr;
    Tensor* w_o = nullptr;
    Tensor* w1 = nullptr;
    Tensor* w2 = nullptr;

    uint32_t model_dim = 0;
    uint32_t hidden_dim = 0;

    void init(Graph& graph, uint32_t model_dim_, uint32_t hidden_dim_) {
        model_dim = model_dim_;
        hidden_dim = hidden_dim_;

        w_q = &graph.tensor({model_dim, model_dim}, true);
        w_k = &graph.tensor({model_dim, model_dim}, true);
        w_v = &graph.tensor({model_dim, model_dim}, true);
        w_o = &graph.tensor({model_dim, model_dim}, true);
        w1 = &graph.tensor({model_dim, hidden_dim}, true);
        w2 = &graph.tensor({hidden_dim, model_dim}, true);
    }

    void init_weights(float base_scale) {
        float attn_scale = base_scale * (1.0f / sqrtf(float(model_dim)));
        float ffn_scale = base_scale * sqrtf(2.0f / float(model_dim));
        float out_scale = base_scale * sqrtf(2.0f / float(hidden_dim));

        w_q->random_init(attn_scale);
        w_k->random_init(attn_scale);
        w_v->random_init(attn_scale);
        w_o->random_init(attn_scale);
        w1->random_init(ffn_scale);
        w2->random_init(out_scale);
    }

    Tensor& forward(Graph& graph, Tensor& input) {
        Tensor& norm_in = graph.rms_norm(input);
        Tensor& q = graph.matmul(norm_in, *w_q);
        Tensor& k = graph.matmul(norm_in, *w_k);
        Tensor& v = graph.matmul(norm_in, *w_v);

        Tensor& attn = graph.causal_attention(q, k, v);
        Tensor& attn_proj = graph.matmul(attn, *w_o);
        Tensor& res1 = graph.residual(input, attn_proj);

        Tensor& norm_ffn = graph.rms_norm(res1);
        Tensor& hidden = graph.matmul(norm_ffn, *w1);
        Tensor& hidden_relu = graph.relu(hidden);
        Tensor& hidden_proj = graph.matmul(hidden_relu, *w2);
        Tensor& out = graph.residual(res1, hidden_proj);
        return out;
    }
};

struct MeshTokenModel {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t vocab_size;
    uint32_t model_dim;
    uint32_t hidden_dim;
    uint32_t num_layers;

    Graph graph;

    Tensor* input_tokens = nullptr;  // [B, S], uint16 in fp16 payload
    Tensor* target_tokens = nullptr; // [B, S], uint16 in fp16 payload
    Tensor* token_emb = nullptr;     // [V, D]
    Tensor* pos_emb = nullptr;       // [S, D]
    Tensor* w_out = nullptr;         // [D, V]

    Tensor* logits = nullptr;        // [B, S, V]
    Tensor* loss = nullptr;          // scalar

    std::vector<CausalAttentionBlock> blocks;

    MeshTokenModel(uint32_t batch_size_,
                   uint32_t seq_len_,
                   uint32_t vocab_size_,
                   uint32_t model_dim_,
                   uint32_t hidden_dim_,
                   uint32_t num_layers_)
        : batch_size(batch_size_),
          seq_len(seq_len_),
          vocab_size(vocab_size_),
          model_dim(model_dim_),
          hidden_dim(hidden_dim_),
          num_layers(num_layers_) {
        build_graph();
    }

    void build_graph() {
        input_tokens = &graph.tensor({batch_size, seq_len});
        target_tokens = &graph.tensor({batch_size, seq_len});

        token_emb = &graph.tensor({vocab_size, model_dim}, true);
        pos_emb = &graph.tensor({seq_len, model_dim}, true);

        Tensor& x_emb = graph.embed(*token_emb, *input_tokens);
        Tensor& x_pos = graph.add_position_embedding(x_emb, *pos_emb, batch_size, seq_len);

        blocks.resize(num_layers);
        Tensor* x = &x_pos;
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init(graph, model_dim, hidden_dim);
            x = &blocks[i].forward(graph, *x);
        }

        Tensor& x_norm = graph.rms_norm(*x);
        w_out = &graph.tensor({model_dim, vocab_size}, true);
        logits = &graph.matmul(x_norm, *w_out, 16, 16);
        loss = &graph.cross_entropy_loss(*logits, *target_tokens);
    }

    void init_weights(uint32_t seed = 42) {
        srand(seed);

        token_emb->random_init(0.08f);
        pos_emb->random_init(0.02f);
        w_out->random_init(0.06f);

        for (auto& block : blocks) {
            block.init_weights(0.2f);
        }
    }
};

void sample_greedy_autoregressive(MeshTokenModel& model,
                                  uint32_t batch_size,
                                  std::vector<uint16_t>& generated_inputs,
                                  std::vector<uint16_t>& scratch_targets,
                                  std::vector<float>& logits_cpu) {
    generated_inputs.assign(batch_size * kSeqLen, kPadToken);
    scratch_targets.assign(batch_size * kSeqLen, kPadToken);

    for (uint32_t b = 0; b < batch_size; ++b) {
        generated_inputs[b * kSeqLen + 0] = kBosToken;
    }

    upload_token_tensor(*model.target_tokens, scratch_targets);

    for (uint32_t pos = 0; pos <= kCoordTokenCount; ++pos) {
        upload_token_tensor(*model.input_tokens, generated_inputs);
        model.graph.eval(false);
        download_float_tensor(*model.logits, logits_cpu);

        for (uint32_t b = 0; b < batch_size; ++b) {
            uint16_t next_tok = kPadToken;

            if (pos < kCoordTokenCount) {
                const float* row = logits_cpu.data() + ((b * kSeqLen + pos) * kVocabSize);
                uint16_t best_tok = kCoordTokenBase;
                float best = row[best_tok];
                for (uint16_t tok = uint16_t(kCoordTokenBase + 1); tok < uint16_t(kCoordTokenBase + kCoordBins); ++tok) {
                    if (row[tok] > best) {
                        best = row[tok];
                        best_tok = tok;
                    }
                }
                next_tok = best_tok;
            } else {
                next_tok = kEosToken;
            }

            generated_inputs[b * kSeqLen + (pos + 1)] = next_tok;
        }
    }
}

} // namespace

void main_llm() {
    evk::ai::initialize();
    printf("=== main_llm: causal autoregressive mesh tokens (CE, BOS/EOS, 128 bins) ===\n");

    auto start = std::chrono::high_resolution_clock::now();

    constexpr uint32_t kBatchSize = 16;
    constexpr uint32_t kModelDim = 32;
    constexpr uint32_t kHiddenDim = 64;
    constexpr uint32_t kLayers = 6;
    constexpr uint32_t kTrainSteps = 3000;
    constexpr uint32_t kLogInterval = 200;
    constexpr float kLearningRate = 2.0e-3f;

    MeshTokenModel model(kBatchSize, kSeqLen, kVocabSize, kModelDim, kHiddenDim, kLayers);
    model.init_weights(42);

    std::mt19937 train_rng(1337);
    auto base = make_cube_triangles();

    struct ValSeed {
        MeshBatch batch;
        std::vector<uint16_t> input_tokens;
        std::vector<uint16_t> target_tokens;
        std::vector<float> first_target_mesh;
    };

    constexpr uint32_t kValSeeds = 3;
    std::vector<ValSeed> val(kValSeeds);

    for (uint32_t s = 0; s < kValSeeds; ++s) {
        std::mt19937 rng_val(9001 + s);
        generate_batch(base, kBatchSize, rng_val, val[s].batch);
        build_token_batch(val[s].batch.target, kBatchSize, val[s].input_tokens, val[s].target_tokens);

        val[s].first_target_mesh.assign(kTrianglesPerMesh * kMeshFeatureDim, 0.0f);
        for (uint32_t i = 0; i < kTrianglesPerMesh * kMeshFeatureDim; ++i) {
            val[s].first_target_mesh[i] = val[s].batch.target[i];
        }
    }

    std::filesystem::path evo_path("output/mesh_val_evolution.obj");
    if (std::filesystem::exists(evo_path)) {
        std::filesystem::remove(evo_path);
    }
    uint32_t evo_vertex_index = 1;
    uint32_t evo_snapshot_count = 0;
    const float mesh_spacing = 2.0f;
    const float row_spacing = mesh_spacing * 2.0f;

    for (uint32_t s = 0; s < kValSeeds; ++s) {
        float y_offset = -float(s) * row_spacing;
        append_obj(evo_path,
                   val[s].first_target_mesh,
                   kTrianglesPerMesh,
                   evo_vertex_index,
                   0.0f,
                   y_offset,
                   -1.0e9f);
    }

    std::vector<uint16_t> train_input_tokens;
    std::vector<uint16_t> train_target_tokens;
    std::vector<float> logits_cpu;

    std::vector<uint16_t> sampled_tokens;
    std::vector<uint16_t> scratch_targets;

    for (uint32_t step = 1; step <= kTrainSteps; ++step) {
        MeshBatch batch;
        generate_batch(base, kBatchSize, train_rng, batch);
        build_token_batch(batch.target, kBatchSize, train_input_tokens, train_target_tokens);

        upload_token_tensor(*model.input_tokens, train_input_tokens);
        upload_token_tensor(*model.target_tokens, train_target_tokens);

        model.graph.eval(true);
        model.graph.step_adam(kLearningRate, 0.9f, 0.98f, 1e-4f);
        evk::ai::SubmitCmd(true);

        if (step == 1 || step % kLogInterval == 0 || step == kTrainSteps) {
            float train_loss = float(model.loss->item());

            float val_loss = 0.0f;

            for (uint32_t s = 0; s < kValSeeds; ++s) {
                upload_token_tensor(*model.input_tokens, val[s].input_tokens);
                upload_token_tensor(*model.target_tokens, val[s].target_tokens);
                model.graph.eval(false);
                val_loss += float(model.loss->item());

                sample_greedy_autoregressive(model, kBatchSize, sampled_tokens, scratch_targets, logits_cpu);

                std::vector<float> pred_first_mesh;
                decode_generated_mesh(sampled_tokens, 0, pred_first_mesh);
                float y_offset = -float(s) * row_spacing;
                append_obj(evo_path,
                           pred_first_mesh,
                           kTrianglesPerMesh,
                           evo_vertex_index,
                           mesh_spacing * float(evo_snapshot_count + 1),
                           y_offset,
                           -1.0e9f);
            }

            val_loss /= float(kValSeeds);

                 printf("step %4u | train_ce %.6f | val_ce %.6f\n",
                     step, train_loss, val_loss);
            fflush(stdout);
            evo_snapshot_count++;
        }
    }

    // Final sample/export from seed 0
    sample_greedy_autoregressive(model, kBatchSize, sampled_tokens, scratch_targets, logits_cpu);

    std::vector<float> pred_first_mesh;
    decode_generated_mesh(sampled_tokens, 0, pred_first_mesh);

    export_obj("output/mesh_target.obj", val[0].first_target_mesh, kTrianglesPerMesh);
    export_obj("output/mesh_pred.obj", pred_first_mesh, kTrianglesPerMesh);

    // Also save a few independent greedy samples (all start at BOS)
    for (uint32_t s = 0; s < 3; ++s) {
        decode_generated_mesh(sampled_tokens, s, pred_first_mesh);
        export_obj(std::filesystem::path("output") / ("mesh_pred_seed" + std::to_string(s) + ".obj"),
                   pred_first_mesh,
                   kTrianglesPerMesh);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("main_llm() took %.4f seconds\n", duration.count());

    evk::ai::shutdown();
}
