#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include <evk_ai.h>

namespace {
constexpr uint32_t kTrianglesPerMesh = 12;
constexpr uint32_t kCoordsPerTriangle = 9;
constexpr uint32_t kMeshFeatureDim = kCoordsPerTriangle;

constexpr uint16_t kPadToken = 0; // also ignore token for CE
constexpr uint16_t kBosToken = 1;
constexpr uint16_t kEosToken = 2;
constexpr uint16_t kCoordTokenBase = 3;
constexpr uint32_t kCoordBins = 128;
constexpr uint32_t kConditionTriangles = 2;
constexpr uint32_t kConditionCoordTokens = kConditionTriangles * kCoordsPerTriangle; // 18

constexpr uint32_t kCoordTokenCount = kTrianglesPerMesh * kCoordsPerTriangle; // 108
constexpr uint32_t kSeqActiveLen = 1 + kCoordTokenCount + 1; // BOS + coords + EOS = 110
constexpr uint32_t kSeqLen = 112; // Smallest tile-aligned length covering 110 active tokens.
constexpr uint32_t kVocabSize = 144; // tile-aligned, uses ids [0..130]

// Covers the full transformed cube support: +/-0.45 translation plus sqrt(3)/2 rotation extent.
constexpr float kCoordMin = -1.32f;
constexpr float kCoordMax = 1.32f;
constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Mat3 {
    float m[9];
};

using Triangle = std::array<float, kCoordsPerTriangle>;
using Mesh = std::vector<Triangle>;

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 scale(const Vec3& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

Vec3 triangle_centroid(const Vec3& a, const Vec3& b, const Vec3& c) {
    return scale(add(add(a, b), c), 1.0f / 3.0f);
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

Mesh make_cube_triangles() {
    Vec3 v0{-0.5f, -0.5f, -0.5f};
    Vec3 v1{ 0.5f, -0.5f, -0.5f};
    Vec3 v2{ 0.5f,  0.5f, -0.5f};
    Vec3 v3{-0.5f,  0.5f, -0.5f};
    Vec3 v4{-0.5f, -0.5f,  0.5f};
    Vec3 v5{ 0.5f, -0.5f,  0.5f};
    Vec3 v6{ 0.5f,  0.5f,  0.5f};
    Vec3 v7{-0.5f,  0.5f,  0.5f};

    auto tri = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        return Triangle{a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z};
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

Mesh make_tetrahedron_triangles() {
    constexpr float s = 0.5f;
    Vec3 v0{ s,  s,  s};
    Vec3 v1{-s, -s,  s};
    Vec3 v2{-s,  s, -s};
    Vec3 v3{ s, -s, -s};

    auto tri = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        return Triangle{a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z};
    };

    auto subdivide_face = [&](const Vec3& a, const Vec3& b, const Vec3& c) {
        Vec3 center = triangle_centroid(a, b, c);
        return std::array<Triangle, 3>{
            tri(a, b, center),
            tri(b, c, center),
            tri(c, a, center),
        };
    };

    Mesh tris;
    tris.reserve(kTrianglesPerMesh);

    auto append_face = [&](const std::array<Triangle, 3>& face_tris) {
        tris.insert(tris.end(), face_tris.begin(), face_tris.end());
    };

    append_face(subdivide_face(v0, v2, v1));
    append_face(subdivide_face(v0, v1, v3));
    append_face(subdivide_face(v0, v3, v2));
    append_face(subdivide_face(v1, v2, v3));

    assert(tris.size() == kTrianglesPerMesh);
    return tris;
}

void apply_transform(Triangle& tri, const Mat3& r, const Vec3& t) {
    for (int i = 0; i < 3; ++i) {
        Vec3 v{tri[i * 3 + 0], tri[i * 3 + 1], tri[i * 3 + 2]};
        v = add(mul(r, v), t);
        tri[i * 3 + 0] = v.x;
        tri[i * 3 + 1] = v.y;
        tri[i * 3 + 2] = v.z;
    }
}

void generate_batch(const std::vector<Mesh>& base_shapes,
                    uint32_t batch_size,
                    std::mt19937& rng,
                    std::vector<float>& target) {
    assert(!base_shapes.empty());
    target.assign(batch_size * kTrianglesPerMesh * kMeshFeatureDim, 0.0f);
    std::uniform_int_distribution<size_t> shape_dist(0, base_shapes.size() - 1);

    for (uint32_t b = 0; b < batch_size; ++b) {
        auto tris = base_shapes[shape_dist(rng)];
        assert(tris.size() == kTrianglesPerMesh);
        Mat3 rot = random_rotation(rng);
        Vec3 translate{
            rand_uniform(rng, -0.45f, 0.45f),
            rand_uniform(rng, -0.45f, 0.45f),
            rand_uniform(rng, -0.45f, 0.45f)
        };

        for (auto& tri : tris) {
            apply_transform(tri, rot, translate);
        }

        for (uint32_t t = 0; t < kTrianglesPerMesh; ++t) {
            const auto& tri = tris[t];
            uint32_t base_idx = (b * kTrianglesPerMesh + t) * kMeshFeatureDim;

            for (uint32_t i = 0; i < kCoordsPerTriangle; ++i) {
                target[base_idx + i] = tri[i];
            }
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

        in_ptr[kSeqActiveLen - 1] = kEosToken;

        // Teacher-forcing target: predict next token.
        for (uint32_t p = 0; p < kSeqActiveLen - 1; ++p) {
            tgt_ptr[p] = in_ptr[p + 1];
        }

        // The first two triangles are provided as conditioning prefix.
        for (uint32_t p = 0; p < kConditionCoordTokens; ++p) {
            tgt_ptr[p] = kPadToken;
        }
    }
}

void upload_token_tensor(Tensor& t,
                         const std::vector<uint16_t>& data,
                         bool submit = true) {
    assert(t.shape.count() == data.size());
    auto& cmd = evk::ai::GetCmd();
    cmd.copy((void*)data.data(), t.buffer, t.shape.count() * sizeof(uint16_t));
    if (submit) {
        evk::ai::SubmitCmd(true);
    }
}

void queue_tensor_download(Tensor& t) {
    t.cpu();
    auto& cmd = evk::ai::GetCmd();
    cmd.copy(t.buffer, t.cpu_buffer, t.shape.count() * sizeof(float16_t));
}

using ParamSnapshot = std::vector<std::vector<float16_t>>;

struct ValSeed {
    std::vector<uint16_t> input_tokens;
    std::vector<uint16_t> target_tokens;
    std::vector<float> first_target_mesh;
};

struct EvalMetrics {
    float val_ce = 0.0f;
    float val_completion_mse = 0.0f;
};

ParamSnapshot capture_params(Graph& graph) {
    ParamSnapshot snapshot(graph.params.size());

    for (Tensor* param : graph.params) {
        param->cpu_download(false);
    }
    evk::ai::SubmitCmd(true);

    for (size_t i = 0; i < graph.params.size(); ++i) {
        Tensor& param = *graph.params[i];
        float16_t* src = param.cpu();
        snapshot[i].assign(src, src + param.shape.count());
    }

    return snapshot;
}

void restore_params(Graph& graph, const ParamSnapshot& snapshot) {
    assert(snapshot.size() == graph.params.size());

    for (size_t i = 0; i < graph.params.size(); ++i) {
        Tensor& param = *graph.params[i];
        const auto& src = snapshot[i];
        assert(src.size() == param.shape.count());

        float16_t* dst = param.cpu();
        std::copy(src.begin(), src.end(), dst);
        param.cpu_upload(false);
    }

    evk::ai::SubmitCmd(true);
}

void seed_condition_prefix_tokens(const std::vector<float>& condition_meshes,
                                  uint32_t batch_size,
                                  std::vector<uint16_t>& generated_inputs) {
    assert(condition_meshes.size() >= size_t(batch_size) * kTrianglesPerMesh * kMeshFeatureDim);
    generated_inputs.assign(batch_size * kSeqLen, kPadToken);

    for (uint32_t b = 0; b < batch_size; ++b) {
        uint16_t* in_ptr = generated_inputs.data() + b * kSeqLen;
        in_ptr[0] = kBosToken;
        in_ptr[kSeqActiveLen - 1] = kEosToken;

        for (uint32_t i = 0; i < kConditionCoordTokens; ++i) {
            uint32_t tri = i / kCoordsPerTriangle;
            uint32_t coord = i % kCoordsPerTriangle;
            uint32_t idx = (b * kTrianglesPerMesh + tri) * kMeshFeatureDim + coord;
            in_ptr[1 + i] = quantize_coord_to_token(condition_meshes[idx]);
        }
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

}

void copy_condition_prefix(const std::vector<float>& condition_mesh,
                           std::vector<float>& mesh_features_io) {
    assert(condition_mesh.size() >= kTrianglesPerMesh * kMeshFeatureDim);
    assert(mesh_features_io.size() >= kTrianglesPerMesh * kMeshFeatureDim);

    std::copy_n(condition_mesh.begin(),
                kConditionTriangles * kMeshFeatureDim,
                mesh_features_io.begin());
}

float completion_mse(const std::vector<float>& pred_mesh,
                     const std::vector<float>& target_mesh) {
    assert(pred_mesh.size() >= kTrianglesPerMesh * kMeshFeatureDim);
    assert(target_mesh.size() >= kTrianglesPerMesh * kMeshFeatureDim);

    double sum_sq = 0.0;
    for (uint32_t tri = kConditionTriangles; tri < kTrianglesPerMesh; ++tri) {
        uint32_t base_idx = tri * kMeshFeatureDim;
        for (uint32_t coord = 0; coord < kCoordsPerTriangle; ++coord) {
            double diff = double(pred_mesh[base_idx + coord] - target_mesh[base_idx + coord]);
            sum_sq += diff * diff;
        }
    }

    constexpr uint32_t completion_coordinate_count =
        (kTrianglesPerMesh - kConditionTriangles) * kCoordsPerTriangle;
    return float(sum_sq / double(completion_coordinate_count));
}

void append_obj(std::ostream& out,
                const std::vector<float>& features,
                uint32_t& vertex_index,
                float x_offset,
                float y_offset) {
    for (uint32_t t = 0; t < kTrianglesPerMesh; ++t) {
        uint32_t base_idx = t * kMeshFeatureDim;
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

enum class AttentionMode {
    Softmax,
    GatedDelta,
    Yoco,
    YocoConv,
};

bool is_yoco(AttentionMode mode) {
    return mode == AttentionMode::Yoco || mode == AttentionMode::YocoConv;
}

// Full-sequence projection tiling; cached single-token projections use their own defaults.
constexpr uint8_t kYocoProjectionTileN = 64;

struct GatedDeltaStepModel;
struct AttentionStepModel;

struct SequenceBlock {
    Tensor* w_conv = nullptr;
    Tensor* w_projection = nullptr;
    Tensor* w_q = nullptr;
    Tensor* w_k = nullptr;
    Tensor* w_v = nullptr;
    Tensor* w_o = nullptr;
    Tensor* w1 = nullptr;
    Tensor* w2 = nullptr;
    uint32_t model_dim = 0;
    uint32_t head_count = 1;
    float rope_base = 10000.0f;
    AttentionMode attention_mode = AttentionMode::Softmax;

    void init(Graph& graph,
              uint32_t model_dim_,
              uint32_t hidden_dim,
              float rope_base_,
              AttentionMode attention_mode_,
              uint32_t head_count_, bool cross_attention = false, uint32_t conv_kernel = 0) {
        model_dim = model_dim_;
        rope_base = rope_base_;
        attention_mode = attention_mode_;
        head_count = head_count_;

        if (conv_kernel) {
            w_conv = &graph.tensor({conv_kernel, model_dim}, true);
        } else if (attention_mode == AttentionMode::GatedDelta) {
            w_projection = &graph.tensor(
                {model_dim, 3u * model_dim + 2u * head_count}, true);
        } else {
            w_q = &graph.tensor({model_dim, model_dim}, true);
            if (!cross_attention) {
                w_k = &graph.tensor({model_dim, model_dim}, true);
                w_v = &graph.tensor({model_dim, model_dim}, true);
            }
        }
        w_o = &graph.tensor({model_dim, model_dim}, true);
        w1 = &graph.tensor({model_dim, hidden_dim * (is_yoco(attention_mode) ? 2u : 1u)}, true);
        w2 = &graph.tensor({hidden_dim, model_dim}, true);
    }

    void init_weights(float weight_stddev, float residual_proj_stddev) {
        if (w_conv) {
            w_conv->random_init(1.0f / std::sqrt(float(w_conv->shape[0])));
        } else if (w_projection) {
            float16_t* data = w_projection->cpu();
            uint32_t stride = w_projection->shape[1];
            std::fill(data, data + w_projection->shape.count(), float16_t(0.0f));
            for (uint32_t segment = 0u; segment < 3u; ++segment) {
                for (uint32_t i = 0u; i < model_dim * model_dim; ++i) {
                    // Preserve Tensor::random_init's exact sampling sequence so
                    // packing Q/K/V does not change a seeded training run.
                    float u1 = float(rand() + 1) / float(RAND_MAX + 1);
                    float u2 = float(rand()) / float(RAND_MAX);
                    float z = sqrtf(-2.0f * logf(u1)) *
                              cosf(2.0f * 3.14159265f * u2);
                    uint32_t row = i / model_dim;
                    uint32_t column = i % model_dim;
                    data[row * stride + segment * model_dim + column] =
                        float16_t(z * weight_stddev);
                }
            }
            w_projection->cpu_upload();
        } else {
            w_q->random_init(weight_stddev);
            if (w_k) w_k->random_init(weight_stddev);
            if (w_v) w_v->random_init(weight_stddev);
        }
        w_o->random_init(residual_proj_stddev);
        w1->random_init(weight_stddev);
        w2->random_init(residual_proj_stddev);
    }

    Tensor& feed_forward(Graph& graph, Tensor& residual) {
        if (is_yoco(attention_mode))
            return graph.swiglu_ffn(residual, *w1, *w2, residual, 32, 1e-4f);
        Tensor& norm = graph.rms_norm(residual);
        Tensor& hidden = graph.matmul_gelu(norm, *w1);
        return graph.matmul_residual(hidden, *w2, residual);
    }

    Tensor& forward(Graph& graph, Tensor& input, Tensor* shared_key = nullptr,
                    Tensor* shared_value = nullptr, uint32_t window = 0) {
        uint8_t tile_n = is_yoco(attention_mode) ? kYocoProjectionTileN : 16;
        Tensor* attn = nullptr;
        if (w_conv) {
            attn = &graph.causal_depthwise_conv(graph.rms_norm(input), *w_conv);
        } else if (attention_mode == AttentionMode::GatedDelta) {
            Tensor& projection = graph.matmul(graph.rms_norm(input), *w_projection);
            Tensor& delta = graph.gated_delta_projected(
                projection, model_dim, head_count, rope_base, -4.0f, 16u);
            attn = &graph.rms_norm(delta);
        } else if (shared_key) {
            Tensor& q = graph.matmul_rope(input, *w_q, rope_base, 1e-4f, tile_n);
            attn = &graph.causal_attention(q, *shared_key, *shared_value, 0.0f, window);
        } else {
            Tensor& norm_in = graph.rms_norm(input);
            Tensor& q_rope = graph.matmul_rope(norm_in, *w_q, rope_base, 0.0f, tile_n);
            Tensor& k = w_k ? graph.matmul_rope(norm_in, *w_k, rope_base, 0.0f, tile_n) : *shared_key;
            Tensor& v = w_v ? graph.matmul(norm_in, *w_v, 16, tile_n) : *shared_value;
            attn = &graph.causal_attention(q_rope, k, v, 0.0f, window);
        }
        Tensor& res1 = graph.matmul_residual(*attn, *w_o, input, 16, tile_n);

        return feed_forward(graph, res1);
    }
};

struct TokenModel {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t vocab_size;
    uint32_t model_dim;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t head_count;
    float rope_base;
    AttentionMode attention_mode;

    Graph graph;

    Tensor* input_tokens = nullptr;  // [B, S], uint16 in fp16 payload
    Tensor* target_tokens = nullptr; // [B, S], uint16 in fp16 payload
    Tensor* token_emb = nullptr;     // [V, D]
    Tensor* w_out = nullptr;         // [D, V]
    Tensor* sampled_token_ids = nullptr; // [B], raw uint16 token ids in fp16 payload storage

    Tensor* logits = nullptr;        // [B, S, V]
    Tensor* loss = nullptr;          // scalar
    Tensor* shared_w_k = nullptr;
    Tensor* shared_w_v = nullptr;
    uint32_t window; // Local attention window or causal convolution kernel size.

    std::vector<SequenceBlock> blocks;
    std::unique_ptr<GatedDeltaStepModel> gated_delta_decoder;
    std::unique_ptr<AttentionStepModel> attention_decoder;

    TokenModel(uint32_t batch_size_,
                   uint32_t seq_len_,
                   uint32_t vocab_size_,
                   uint32_t model_dim_,
                   uint32_t hidden_dim_,
                   uint32_t num_layers_,
                   float rope_base_,
                   AttentionMode attention_mode_,
                   uint32_t head_count_, uint32_t window_ = 32u)
        : batch_size(batch_size_),
          seq_len(seq_len_),
          vocab_size(vocab_size_),
          model_dim(model_dim_),
          hidden_dim(hidden_dim_),
          num_layers(num_layers_),
          head_count(head_count_),
          rope_base(rope_base_),
          attention_mode(attention_mode_), window(window_) {
        build_graph();
    }

    ~TokenModel();

    void build_graph() {
        input_tokens = &graph.tensor({batch_size, seq_len});
        target_tokens = &graph.tensor({batch_size, seq_len});

        token_emb = &graph.tensor({vocab_size, model_dim}, true);
        Tensor& x_emb = graph.embed(*token_emb, *input_tokens);
        blocks.resize(num_layers);
        Tensor* x = &x_emb;
        Tensor* shared_key = nullptr;
        Tensor* shared_value = nullptr;
        for (uint32_t i = 0; i < num_layers; ++i) {
            bool cross = is_yoco(attention_mode) && i >= num_layers / 2u;
            if (cross && i == num_layers / 2u) {
                Tensor& memory = graph.rms_norm(*x);
                shared_w_k = &graph.tensor({model_dim, model_dim}, true);
                shared_w_v = &graph.tensor({model_dim, model_dim}, true);
                shared_key = &graph.matmul_rope(memory, *shared_w_k, rope_base, 0.0f, kYocoProjectionTileN);
                shared_value = &graph.matmul(memory, *shared_w_v, 16, kYocoProjectionTileN);
            }
            blocks[i].init(graph,
                           model_dim,
                           hidden_dim,
                           rope_base,
                           attention_mode,
                           head_count, cross,
                           attention_mode == AttentionMode::YocoConv && !cross ? window : 0u);
            x = &blocks[i].forward(graph, *x, shared_key, shared_value,
                is_yoco(attention_mode) && !cross ? window : 0u);
        }

        Tensor& x_norm = graph.rms_norm(*x);
        w_out = &graph.tensor({model_dim, vocab_size}, true);
        sampled_token_ids = &graph.tensor({batch_size});
        logits = &graph.matmul(x_norm, *w_out, 16, 16);
        loss = &graph.cross_entropy_loss(*logits, *target_tokens);
    }

    void init_weights(uint32_t seed = 42) {
        srand(seed);

        constexpr float transformer_weight_stddev = 0.02f;
        float residual_proj_stddev = transformer_weight_stddev / sqrtf(2.0f * float(num_layers));
        token_emb->random_init(transformer_weight_stddev);
        w_out->random_init(transformer_weight_stddev);

        for (auto& block : blocks) {
            block.init_weights(transformer_weight_stddev, residual_proj_stddev);
        }
        if (shared_w_k) shared_w_k->random_init(transformer_weight_stddev);
        if (shared_w_v) shared_w_v->random_init(transformer_weight_stddev);
    }
};

struct GatedDeltaStepModel {
    static constexpr uint32_t kTileRows = 16u;

    Graph graph;
    Tensor* input_tokens = nullptr;
    Tensor* input_positions = nullptr;
    Tensor* logits = nullptr;
    Tensor* sampled_token_ids = nullptr;
    std::vector<Tensor*> states;

    explicit GatedDeltaStepModel(TokenModel& source) {
        assert(source.attention_mode == AttentionMode::GatedDelta);
        assert(source.batch_size <= kTileRows);
        input_tokens = &graph.tensor({1u, kTileRows});
        input_positions = &graph.tensor({1u, kTileRows});
        Tensor* x = &graph.embed(*source.token_emb, *input_tokens);
        uint32_t head_dim = source.model_dim / source.head_count;
        states.reserve(source.num_layers);
        for (uint32_t i = 0; i < source.num_layers; ++i) {
            SequenceBlock& block = source.blocks[i];
            Tensor& state = graph.tensor(
                {source.batch_size, source.head_count, head_dim, head_dim});
            states.push_back(&state);
            Tensor& projection = graph.matmul(
                graph.rms_norm(*x), *block.w_projection);
            Tensor& mixed = graph.gated_delta_projected_step(
                projection, *input_positions, state,
                source.model_dim, source.head_count);
            Tensor& residual = graph.matmul_residual(
                graph.rms_norm(mixed), *block.w_o, *x);
            Tensor& hidden = graph.matmul_gelu(
                graph.rms_norm(residual), *block.w1);
            x = &graph.matmul_residual(hidden, *block.w2, residual);
        }
        logits = &graph.matmul(
            graph.rms_norm(*x), *source.w_out, 16, 16);
        sampled_token_ids = &graph.tensor({source.batch_size});
    }

    void reset_state() {
        for (Tensor* state : states) {
            evk::ai::zero(*state, false);
        }
        evk::ai::GetCmd().computeBarrier();
    }
};

// Both decoders reuse the training weights; only inference activations/caches are owned here.
struct AttentionStepModel {
    Graph graph;
    Tensor* input_tokens = nullptr;
    Tensor* logits = nullptr;
    Tensor* sampled_token_ids = nullptr;
    uint32_t position = 0;
    size_t prefill_end = 0;
    uint64_t cache_bytes = 0;

    std::pair<Tensor*, Tensor*> cache(Tensor& k, Tensor& v, uint32_t capacity, float base) {
        Tensor& kc = graph.tensor({k.shape[1], capacity, k.shape[2]});
        Tensor& vc = graph.tensor(kc.shape);
        kc.forward_fn = [this, &k, &v, &kc, &vc, base]() {
            evk::ai::attention_cache_append(k, v, kc, vc, position, base);
        };
        cache_bytes += 2u * uint64_t(kc.shape.count()) * sizeof(float16_t);
        return {&kc, &vc};
    }

    explicit AttentionStepModel(TokenModel& source) {
        assert(source.batch_size == 16u && source.attention_mode != AttentionMode::GatedDelta);
        input_tokens = &graph.tensor({1u, source.batch_size});
        Tensor* x = &graph.embed(*source.token_emb, *input_tokens);
        Tensor* shared_key = nullptr;
        Tensor* shared_value = nullptr;
        // Layer temporaries are dead before the next layer uses the same role.
        // This decoder never runs backward; caches remain independently owned.
        Tensor *norm_storage = nullptr, *raw_query_storage = nullptr, *query_storage = nullptr;
        Tensor *key_storage = nullptr, *value_storage = nullptr, *mixed_storage = nullptr;
        Tensor* residual_storage = nullptr;
        std::vector<Tensor*> ff_storage, ff_workspace_storage;
        auto reuse = [](Tensor& tensor, Tensor*& storage) {
            if (!storage) storage = &tensor;
            else {
                assert(tensor.shape == storage->shape);
                tensor.buffer = storage->buffer;
            }
        };
        auto reuse_range = [&](auto& tensors, size_t begin, std::vector<Tensor*>& storage) {
            if (storage.empty()) storage.resize(tensors.size() - begin, nullptr);
            assert(storage.size() == tensors.size() - begin);
            for (size_t j = 0; j < storage.size(); ++j) reuse(*tensors[begin + j], storage[j]);
        };
        for (uint32_t i = 0; i < source.num_layers; ++i) {
            SequenceBlock& block = source.blocks[i];
            if (is_yoco(source.attention_mode) && i == source.num_layers / 2u) {
                Tensor& memory = graph.rms_norm(*x);
                Tensor& k = graph.matmul(memory, *source.shared_w_k);
                Tensor& v = graph.matmul(memory, *source.shared_w_v);
                auto caches = cache(k, v, source.seq_len, source.rope_base);
                shared_key = caches.first;
                shared_value = caches.second;
                prefill_end = graph.nodes.size();
            }
            Tensor& norm = graph.rms_norm(*x);
            reuse(norm, norm_storage);
            Tensor* mixed_output = nullptr;
            if (block.w_conv) {
                Tensor& state = graph.tensor({source.batch_size, block.w_conv->shape[0], source.model_dim});
                cache_bytes += uint64_t(state.shape.count()) * sizeof(float16_t);
                Tensor& mixed = graph.tensor(norm.shape);
                reuse(mixed, mixed_storage);
                mixed.forward_fn = [this, &norm, weight = block.w_conv, &state, &mixed]() {
                    evk::ai::causal_depthwise_conv_step(norm, *weight, state, mixed, position);
                };
                mixed_output = &mixed;
            } else {
                Tensor& raw_query = graph.matmul(norm, *block.w_q);
                reuse(raw_query, raw_query_storage);
                Tensor& q = graph.tensor(raw_query.shape);
                reuse(q, query_storage);
                q.forward_fn = [this, &raw_query, &q, base = source.rope_base]() {
                    evk::ai::rope(raw_query, q, 1u, raw_query.shape[1], raw_query.shape[2], base, 0.0f, float(position));
                };
                Tensor* key = shared_key;
                Tensor* value = shared_value;
                if (block.w_k) {
                    Tensor& k = graph.matmul(norm, *block.w_k);
                    Tensor& v = graph.matmul(norm, *block.w_v);
                    reuse(k, key_storage);
                    reuse(v, value_storage);
                    auto caches = cache(k, v,
                        source.attention_mode == AttentionMode::Yoco ? source.window : source.seq_len,
                        source.rope_base);
                    key = caches.first;
                    value = caches.second;
                }
                Tensor& mixed = graph.tensor(q.shape);
                reuse(mixed, mixed_storage);
                mixed.forward_fn = [this, &q, key, value, &mixed]() {
                    evk::ai::cached_attention(q, *key, *value, mixed, position);
                };
                mixed_output = &mixed;
            }
            Tensor& residual = graph.matmul_residual(*mixed_output, *block.w_o, *x);
            reuse(residual, residual_storage);
            size_t ff_begin = graph.nodes.size(), workspace_begin = graph.workspaces.size();
            x = &block.feed_forward(graph, residual);
            reuse_range(graph.nodes, ff_begin, ff_storage);
            reuse_range(graph.workspaces, workspace_begin, ff_workspace_storage);
        }
        logits = &graph.matmul(graph.rms_norm(*x), *source.w_out, 16, 16);
        sampled_token_ids = &graph.tensor({source.batch_size});
    }

    void eval(bool prefill = false) {
        size_t end = prefill && prefill_end ? prefill_end : graph.nodes.size();
        for (size_t i = 0; i < end; ++i)
            if (graph.nodes[i]->forward_fn) graph.nodes[i]->forward_fn();
    }
};

TokenModel::~TokenModel() = default;

void sample_attention_autoregressive(TokenModel& model,
    const std::vector<float>& condition_meshes, std::vector<uint16_t>& generated_inputs) {
    seed_condition_prefix_tokens(condition_meshes, model.batch_size, generated_inputs);
    if (!model.attention_decoder) model.attention_decoder = std::make_unique<AttentionStepModel>(model);
    AttentionStepModel& decoder = *model.attention_decoder;
    Tensor prefix({kConditionCoordTokens + 1u, model.batch_size});
    Tensor generated({kCoordTokenCount, model.batch_size});
    for (uint32_t t = 0; t <= kConditionCoordTokens; ++t)
        for (uint32_t b = 0; b < model.batch_size; ++b)
            prefix.cpu()[t * model.batch_size + b].value = generated_inputs[b * kSeqLen + t];
    prefix.cpu_upload(false);
    uint64_t row_bytes = model.batch_size * sizeof(float16_t);
    for (uint32_t t = 0; t < kCoordTokenCount; ++t) {
        decoder.position = t;
        if (t <= kConditionCoordTokens)
            evk::ai::GetCmd().copy(prefix.buffer, decoder.input_tokens->buffer,
                row_bytes, uint64_t(t) * row_bytes);
        decoder.eval(t < kConditionCoordTokens);
        if (t >= kConditionCoordTokens) {
            evk::ai::greedy_sample_rows(*decoder.logits, *decoder.sampled_token_ids,
                kCoordTokenBase, uint16_t(kCoordBins));
            evk::ai::GetCmd().copy(decoder.sampled_token_ids->buffer, generated.buffer,
                row_bytes, 0u, uint64_t(t) * row_bytes);
            evk::ai::GetCmd().copy(decoder.sampled_token_ids->buffer, decoder.input_tokens->buffer, row_bytes);
        }
    }
    generated.cpu_download();
    for (uint32_t t = kConditionCoordTokens; t < kCoordTokenCount; ++t)
        for (uint32_t b = 0; b < model.batch_size; ++b)
            generated_inputs[b * kSeqLen + t + 1u] = generated.cpu()[t * model.batch_size + b].value;
}

void sample_gated_delta_autoregressive(
    TokenModel& model,
    const std::vector<float>& condition_meshes,
    std::vector<uint16_t>& generated_inputs) {
    const uint32_t batch_size = model.batch_size;
    seed_condition_prefix_tokens(condition_meshes, batch_size, generated_inputs);

    if (!model.gated_delta_decoder) {
        model.gated_delta_decoder = std::make_unique<GatedDeltaStepModel>(model);
    }
    GatedDeltaStepModel& step_model = *model.gated_delta_decoder;
    step_model.reset_state();
    Tensor prefix_tokens({kConditionCoordTokens + 1u,
                          GatedDeltaStepModel::kTileRows});
    Tensor position_rows({kCoordTokenCount, GatedDeltaStepModel::kTileRows});
    Tensor generated_rows({kSeqLen, GatedDeltaStepModel::kTileRows});
    for (uint32_t position = 0u; position <= kConditionCoordTokens; ++position) {
        for (uint32_t batch = 0u; batch < batch_size; ++batch) {
            prefix_tokens.cpu()[position * GatedDeltaStepModel::kTileRows + batch].value =
                generated_inputs[batch * kSeqLen + position];
        }
    }
    for (uint32_t position = 0u; position < kCoordTokenCount; ++position) {
        for (uint32_t batch = 0u; batch < batch_size; ++batch) {
            position_rows.cpu()[position * GatedDeltaStepModel::kTileRows + batch].value =
                uint16_t(position);
        }
    }
    prefix_tokens.cpu_upload(false);
    position_rows.cpu_upload(false);

    constexpr uint64_t row_bytes =
        GatedDeltaStepModel::kTileRows * sizeof(float16_t);
    for (uint32_t position = 0u; position < kCoordTokenCount; ++position) {
        evk::ai::GetCmd().copy(position_rows.buffer,
                               step_model.input_positions->buffer,
                               row_bytes,
                               uint64_t(position) * row_bytes);
        if (position <= kConditionCoordTokens) {
            evk::ai::GetCmd().copy(prefix_tokens.buffer,
                                   step_model.input_tokens->buffer,
                                   row_bytes,
                                   uint64_t(position) * row_bytes);
        }
        step_model.graph.eval(false, false, false);

        if (position >= kConditionCoordTokens) {
            evk::ai::greedy_sample_rows(*step_model.logits,
                                        *step_model.sampled_token_ids,
                                        kCoordTokenBase,
                                        uint16_t(kCoordBins));
            evk::ai::GetCmd().copy(step_model.sampled_token_ids->buffer,
                                   generated_rows.buffer,
                                   batch_size * sizeof(float16_t),
                                   0u,
                                   uint64_t(position + 1u) * row_bytes);
            evk::ai::GetCmd().copy(step_model.sampled_token_ids->buffer,
                                   step_model.input_tokens->buffer,
                                   batch_size * sizeof(float16_t));
        }
    }
    generated_rows.cpu_download();
    for (uint32_t position = kConditionCoordTokens + 1u;
         position <= kCoordTokenCount; ++position) {
        for (uint32_t batch = 0u; batch < batch_size; ++batch) {
            generated_inputs[batch * kSeqLen + position] =
                generated_rows.cpu()[position * GatedDeltaStepModel::kTileRows + batch].value;
        }
    }
}

void sample_autoregressive(TokenModel& model,
                           const std::vector<float>& condition_meshes,
                           std::vector<uint16_t>& generated_inputs) {
    if (model.attention_mode == AttentionMode::GatedDelta) {
        sample_gated_delta_autoregressive(model, condition_meshes, generated_inputs);
        return;
    }
    sample_attention_autoregressive(model, condition_meshes, generated_inputs);
}

void sample_attention_reference(TokenModel& model,
    const std::vector<float>& condition_meshes, std::vector<uint16_t>& generated_inputs,
    std::vector<uint16_t>& scratch_targets) {
    const uint32_t batch_size = model.batch_size;
    scratch_targets.assign(batch_size * kSeqLen, kPadToken);

    seed_condition_prefix_tokens(condition_meshes, batch_size, generated_inputs);

    upload_token_tensor(*model.target_tokens, scratch_targets, false);

    for (uint32_t pos = kConditionCoordTokens; pos < kCoordTokenCount; ++pos) {
        upload_token_tensor(*model.input_tokens, generated_inputs, false);
        model.graph.eval(false, false, false);
        evk::ai::greedy_sample(*model.logits,
                               *model.sampled_token_ids,
                               pos,
                               kCoordTokenBase,
                               uint16_t(kCoordBins));
        queue_tensor_download(*model.sampled_token_ids);
        evk::ai::SubmitCmd(true);

        float16_t* sampled_ptr = model.sampled_token_ids->cpu();

        for (uint32_t b = 0; b < batch_size; ++b) {
            generated_inputs[b * kSeqLen + (pos + 1)] = sampled_ptr[b].value;
        }
    }
}

EvalMetrics evaluate_model(TokenModel& model,
                           const std::vector<ValSeed>& val,
                           const std::vector<float>& sample_condition_meshes,
                           std::vector<uint16_t>& sampled_tokens) {
    EvalMetrics metrics;

    for (const auto& seed : val) {
        upload_token_tensor(*model.input_tokens, seed.input_tokens, false);
        upload_token_tensor(*model.target_tokens, seed.target_tokens, false);
        model.graph.eval(false, false, false);
        queue_tensor_download(*model.loss);
        evk::ai::SubmitCmd(true);
        metrics.val_ce += float(model.loss->cpu()[0]);
    }
    metrics.val_ce /= float(val.size());

    sample_autoregressive(model, sample_condition_meshes, sampled_tokens);

    std::vector<float> pred_mesh;
    metrics.val_completion_mse = 0.0f;
    for (size_t s = 0; s < val.size(); ++s) {
        decode_generated_mesh(sampled_tokens, uint32_t(s), pred_mesh);
        metrics.val_completion_mse += completion_mse(pred_mesh, val[s].first_target_mesh);
    }
    metrics.val_completion_mse /= float(val.size());

    return metrics;
}

struct ExperimentResult {
    std::string name;
    uint64_t parameters = 0;
    uint32_t best_step = 0;
    EvalMetrics best;
    double train_seconds = 0.0;
    double train_update_seconds = 0.0;
    double validation_seconds = 0.0;
    uint32_t validation_runs = 0;
    double decode_ms = 0.0;
    uint64_t inference_state_bytes = 0;
    bool inference_state_grows_with_sequence = false;
    uint64_t training_tensor_bytes = 0;
    uint64_t inference_tensor_bytes = 0;
    double cached_logit_rmse = std::nan("");
    double reference_decode_ms = std::nan("");
    double reference_completion_mse = std::nan("");
};

uint64_t graph_tensor_bytes(const Graph& graph) {
    uint64_t bytes = 0;
    std::unordered_map<uint64_t, uint64_t> buffers;
    auto count = [&](Tensor& tensor, uint64_t elements) {
        auto& size = buffers[tensor.buffer.GetReference()];
        size = (std::max)(size, elements * sizeof(float16_t));
    };
    for (const auto& tensor : graph.nodes) {
        count(*tensor, tensor->shape.count());
        if (tensor->grad_tensor) count(*tensor->grad_tensor, tensor->shape.count());
    }
    for (const auto& tensor : graph.workspaces)
        count(*tensor, tensor->shape.count());
    // Adam has two FP16 moment buffers per parameter; exclude CPU staging and Vulkan overhead.
    for (const auto& [parameter, state] : graph.adam_states)
        bytes += uint64_t(parameter->shape.count()) * 2u * sizeof(float16_t);
    for (const auto& buffer : graph.scratch)
        if (buffer.tensor) count(*buffer.tensor, buffer.capacity);
    for (const auto& [buffer, size] : buffers) bytes += size;
    return bytes;
}

double check_cached_logits(TokenModel& model, const ValSeed& val) {
    if (model.attention_mode == AttentionMode::GatedDelta) return std::nan("");
    upload_token_tensor(*model.input_tokens, val.input_tokens, false);
    upload_token_tensor(*model.target_tokens, val.target_tokens, false);
    model.graph.eval(false, false, false);
    model.logits->cpu_download();
    if (!model.attention_decoder) model.attention_decoder = std::make_unique<AttentionStepModel>(model);
    auto& decoder = *model.attention_decoder;
    double squared_error = 0.0;
    double reference_squared = 0.0, reference_ce = 0.0, cached_ce = 0.0;
    uint32_t predicted_rows = 0, matching_tokens = 0;
    uint32_t worst_position = 0;
    bool worst_supervised = false;
    double supervised_squared_error = 0.0;
    size_t supervised_count = 0;
    float worst_reference = 0.0f, worst_actual = 0.0f;
    float max_error = 0.0f;
    size_t count = 0;
    for (uint32_t t = 0; t < kSeqActiveLen; ++t) {
        decoder.position = t;
        for (uint32_t b = 0; b < model.batch_size; ++b)
            decoder.input_tokens->cpu()[b].value = val.input_tokens[b * kSeqLen + t];
        decoder.input_tokens->cpu_upload();
        decoder.eval(t < kConditionCoordTokens);
        if (t < kConditionCoordTokens) continue;
        decoder.logits->cpu_download();
        for (uint32_t b = 0; b < model.batch_size; ++b)
            for (uint32_t v = 0; v < model.vocab_size; ++v) {
                float reference = float(model.logits->cpu()[(b * kSeqLen + t) * model.vocab_size + v]);
                float actual = float(decoder.logits->cpu()[b * model.vocab_size + v]);
                float error = std::abs(reference - actual);
                if (!std::isfinite(error)) throw std::runtime_error("Non-finite cached attention logits");
                if (error > max_error) {
                    max_error = error; worst_reference = reference; worst_actual = actual;
                    worst_position = t;
                    worst_supervised = val.target_tokens[b * kSeqLen + t] != kPadToken;
                }
                squared_error += double(error) * error;
                if (val.target_tokens[b * kSeqLen + t] != kPadToken) {
                    supervised_squared_error += double(error) * error;
                    ++supervised_count;
                }
                reference_squared += double(reference) * reference;
                ++count;
            }
        for (uint32_t b = 0; b < model.batch_size; ++b) {
            uint16_t target = val.target_tokens[b * kSeqLen + t];
            if (target == kPadToken) continue;
            const float16_t* reference = model.logits->cpu() + (b * kSeqLen + t) * model.vocab_size;
            const float16_t* actual = decoder.logits->cpu() + b * model.vocab_size;
            auto ce = [&](const float16_t* row) {
                float maximum = -1e30f;
                for (uint32_t v = 0; v < model.vocab_size; ++v) maximum = (std::max)(maximum, float(row[v]));
                double sum = 0.0;
                for (uint32_t v = 0; v < model.vocab_size; ++v) sum += std::exp(double(float(row[v]) - maximum));
                return std::log(sum) + maximum - float(row[target]);
            };
            reference_ce += ce(reference);
            cached_ce += ce(actual);
            uint32_t ri = 0, ai = 0;
            for (uint32_t v = 1; v < model.vocab_size; ++v) {
                if (float(reference[v]) > float(reference[ri])) ri = v;
                if (float(actual[v]) > float(actual[ai])) ai = v;
            }
            matching_tokens += ri == ai;
            ++predicted_rows;
        }
    }
    double rmse = std::sqrt(squared_error / double(count));
    printf("cached/full logits | max_error %.6f (%.6f vs %.6f) | rmse %.6f | relative_rms %.6f | CE %.6f vs %.6f | argmax %u/%u\n",
        max_error, worst_reference, worst_actual, rmse, std::sqrt(squared_error / reference_squared),
        reference_ce / predicted_rows, cached_ce / predicted_rows, matching_tokens, predicted_rows);
    printf("cached/full supervised_rmse %.6f | worst position %u (%s target)\n",
        std::sqrt(supervised_squared_error / double(supervised_count)), worst_position,
        worst_supervised ? "supervised" : "ignored");
    fflush(stdout);
    if (rmse > 0.02 || std::abs(cached_ce - reference_ce) / predicted_rows > 0.005)
        throw std::runtime_error("Cached/full sequence logits disagree");
    return rmse;
}

void save_checkpoint(const TokenModel& model, const ParamSnapshot& parameters,
                     const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    // Versioned header followed by FP16 parameters in graph registration order.
    const uint32_t header[] = {0x45564b4cu, 1u, uint32_t(model.attention_mode),
        model.model_dim, model.hidden_dim, model.num_layers, model.head_count,
        model.vocab_size, model.window, uint32_t(parameters.size())};
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const auto& parameter : parameters) {
        uint32_t count = uint32_t(parameter.size());
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(parameter.data()), count * sizeof(float16_t));
    }
    if (!file) throw std::runtime_error("Failed to write model checkpoint");
}

void initialize_model(TokenModel& model, uint32_t seed, const std::string& checkpoint) {
    if (checkpoint.empty()) { model.init_weights(seed); return; }
    std::ifstream file(checkpoint, std::ios::binary);
    uint32_t header[10] = {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    const uint32_t expected[] = {0x45564b4cu, 1u, uint32_t(model.attention_mode),
        model.model_dim, model.hidden_dim, model.num_layers, model.head_count,
        model.vocab_size, model.window, uint32_t(model.graph.params.size())};
    if (!file || !std::equal(std::begin(header), std::end(header), std::begin(expected)))
        throw std::runtime_error("Checkpoint architecture does not match selected model");
    for (Tensor* parameter : model.graph.params) {
        uint32_t count = 0;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (count != parameter->shape.count()) throw std::runtime_error("Invalid checkpoint parameter shape");
        file.read(reinterpret_cast<char*>(parameter->cpu()), count * sizeof(float16_t));
        if (!file) throw std::runtime_error("Truncated checkpoint");
        parameter->cpu_upload(false);
    }
    evk::ai::SubmitCmd(true);
}

uint64_t parameter_count(const Graph& graph) {
    uint64_t count = 0;
    for (const Tensor* parameter : graph.params) {
        count += parameter->shape.count();
    }
    return count;
}

ExperimentResult train_experiment(
    TokenModel& model,
    const char* name,
    uint32_t train_steps,
    uint32_t log_interval,
    float learning_rate,
    const std::vector<Mesh>& base_shapes,
    const std::vector<ValSeed>& val,
    const std::vector<float>& sample_condition_meshes,
    const std::filesystem::path& output_dir) {
    ExperimentResult result;
    result.name = name;
    result.parameters = parameter_count(model.graph);
    printf("\n[%s] parameters %llu\n",
           name,
           static_cast<unsigned long long>(result.parameters));

    std::filesystem::path evolution_path =
        output_dir / (std::string(name) + "_mesh_val_evolution.obj");
    std::filesystem::path curve_path =
        output_dir / (std::string(name) + "_training_curve.csv");
    std::ofstream evolution_obj(evolution_path);
    std::ofstream curve_csv(curve_path);
    curve_csv.precision(9);
    curve_csv << "step,train_ce,val_ce,completion_mse,"
                 "cumulative_update_ms,cumulative_validation_ms\n";
    uint32_t evolution_vertex_index = 1;
    uint32_t evolution_snapshot_count = 0;
    constexpr float mesh_spacing = 2.0f;
    constexpr float row_spacing = mesh_spacing * 2.0f;
    for (uint32_t seed = 0; seed < val.size(); ++seed) {
        append_obj(evolution_obj,
                   val[seed].first_target_mesh,
                   evolution_vertex_index,
                   0.0f,
                   -float(seed) * row_spacing);
    }
    evolution_obj.flush();

    std::mt19937 train_rng(1337);
    std::vector<uint16_t> train_input_tokens;
    std::vector<uint16_t> train_target_tokens;
    std::vector<uint16_t> sampled_tokens;
    std::vector<uint16_t> scratch_targets;
    std::vector<float> train_meshes;
    ParamSnapshot best_params;
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step = 1; step <= train_steps; ++step) {
        auto update_start = std::chrono::high_resolution_clock::now();
        bool should_log = step == 1 || step % log_interval == 0 || step == train_steps;
        generate_batch(base_shapes, model.batch_size, train_rng, train_meshes);
        build_token_batch(train_meshes,
                          model.batch_size,
                          train_input_tokens,
                          train_target_tokens);
        upload_token_tensor(*model.input_tokens, train_input_tokens, false);
        upload_token_tensor(*model.target_tokens, train_target_tokens, false);
        model.graph.eval(true, false, false);
        model.graph.step_adam(learning_rate, 0.9f, 0.98f, 1e-4f);
        if (should_log) {
            queue_tensor_download(*model.loss);
        }
        evk::ai::SubmitCmd(should_log);
        auto update_end = std::chrono::high_resolution_clock::now();
        result.train_update_seconds +=
            std::chrono::duration<double>(update_end - update_start).count();
        if (!should_log) {
            continue;
        }

        float train_ce = float(model.loss->cpu()[0]);
        auto validation_start = std::chrono::high_resolution_clock::now();
        EvalMetrics metrics = evaluate_model(model,
                                             val,
                                             sample_condition_meshes,
                                             sampled_tokens);
        if (!std::isfinite(train_ce) || !std::isfinite(metrics.val_ce) ||
            !std::isfinite(metrics.val_completion_mse))
            throw std::runtime_error(std::string(name) + " produced non-finite training/validation metrics");
        auto validation_end = std::chrono::high_resolution_clock::now();
        result.validation_seconds +=
            std::chrono::duration<double>(validation_end - validation_start).count();
        ++result.validation_runs;
        curve_csv << step << ","
                  << train_ce << ","
                  << metrics.val_ce << ","
                  << metrics.val_completion_mse << ","
                  << result.train_update_seconds * 1000.0 << ","
                  << result.validation_seconds * 1000.0 << "\n";
        curve_csv.flush();
        if (best_params.empty() ||
            metrics.val_completion_mse < result.best.val_completion_mse) {
            best_params = capture_params(model.graph);
            result.best_step = step;
            result.best = metrics;
        }

        std::vector<float> predicted_mesh;
        for (uint32_t seed = 0; seed < val.size(); ++seed) {
            decode_generated_mesh(sampled_tokens, seed, predicted_mesh);
            copy_condition_prefix(val[seed].first_target_mesh, predicted_mesh);
            append_obj(evolution_obj,
                       predicted_mesh,
                       evolution_vertex_index,
                       mesh_spacing * float(evolution_snapshot_count + 1),
                       -float(seed) * row_spacing);
        }
        evolution_obj.flush();
        ++evolution_snapshot_count;

        printf("[%s] step %5u | train_ce %.6f | val_ce %.6f | val_completion_mse %.6f\n",
               name,
               step,
               train_ce,
               metrics.val_ce,
               metrics.val_completion_mse);
        fflush(stdout);
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    result.train_seconds = std::chrono::duration<double>(train_end - train_start).count();
    if (best_params.empty()) {
        auto start = std::chrono::high_resolution_clock::now();
        result.best = evaluate_model(model, val, sample_condition_meshes, sampled_tokens);
        result.validation_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        result.validation_runs = 1u;
        best_params = capture_params(model.graph);
    }
    save_checkpoint(model, capture_params(model.graph), output_dir / (std::string(name) + "_final.bin"));
    save_checkpoint(model, best_params, output_dir / (std::string(name) + "_best.bin"));
    restore_params(model.graph, best_params);
    result.cached_logit_rmse = check_cached_logits(model, val.front());
    if (model.attention_mode != AttentionMode::GatedDelta) {
        result.reference_completion_mse = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        sample_attention_reference(model, sample_condition_meshes, sampled_tokens, scratch_targets);
        result.reference_decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        std::vector<float> mesh;
        for (uint32_t b = 0; b < val.size(); ++b) {
            decode_generated_mesh(sampled_tokens, b, mesh);
            result.reference_completion_mse += completion_mse(mesh, val[b].first_target_mesh) / val.size();
        }
        printf("[%s] full-recompute decode_ms: %.3f ms | mesh_mse %.6f\n",
            name, result.reference_decode_ms, result.reference_completion_mse);
    }
    result.training_tensor_bytes = graph_tensor_bytes(model.graph);
    if (model.attention_decoder) {
        result.inference_state_bytes = model.attention_decoder->cache_bytes / model.batch_size;
        result.inference_tensor_bytes = graph_tensor_bytes(model.attention_decoder->graph) +
            result.parameters * sizeof(float16_t);
        result.inference_state_grows_with_sequence = true;
    } else if (model.gated_delta_decoder) {
        for (const Tensor* state : model.gated_delta_decoder->states)
            result.inference_state_bytes += uint64_t(state->shape.count()) * sizeof(float16_t) / model.batch_size;
        result.inference_tensor_bytes = graph_tensor_bytes(model.gated_delta_decoder->graph) +
            result.parameters * sizeof(float16_t);
    }

    std::array<double, 5> decode_times;
    for (double& time : decode_times) {
        auto start = std::chrono::high_resolution_clock::now();
        sample_autoregressive(model, sample_condition_meshes, sampled_tokens);
        time = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }
    std::sort(decode_times.begin(), decode_times.end());
    result.decode_ms = decode_times[decode_times.size() / 2u];

    double update_ms =
        result.train_update_seconds * 1000.0 / double((std::max)(1u, train_steps));
    printf("[%s] train_update_ms: %.3f ms/update\n",
           name,
           update_ms);
    printf("[%s] validation_ms: %.3f ms\n",
           name,
           result.validation_seconds * 1000.0);
    printf("[%s] validation_mean_ms: %.3f ms (%u runs)\n",
           name,
           result.validation_seconds * 1000.0 / double((std::max)(1u, result.validation_runs)),
           result.validation_runs);
    printf("[%s] decode_ms: %.3f ms\n", name, result.decode_ms);

    printf("[%s] restored_best | step %5u | val_ce %.6f | val_completion_mse %.6f | decode_ms %.3f\n",
           name,
           result.best_step,
           result.best.val_ce,
           result.best.val_completion_mse,
           result.decode_ms);
    fflush(stdout);
    return result;
}

uint32_t parse_uint_arg(int argc, char** argv, const char* flag, uint32_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            try {
                return uint32_t(std::stoul(argv[i + 1]));
            } catch (...) {
                return fallback;
            }
        }
    }
    return fallback;
}

std::string parse_string_arg(int argc, char** argv, const char* flag,
                             const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return argv[i + 1];
        }
    }
    return fallback;
}

} // namespace

void main_llm(int argc, char** argv) {
    constexpr uint32_t kBatchSize = 16;
    constexpr uint32_t kModelDim = 256;
    constexpr uint32_t kLayerCount = 8;
    constexpr uint32_t kAttentionHiddenDim = 512;
    constexpr uint32_t kYocoHiddenDim = 400;
    uint32_t gated_delta_heads = parse_uint_arg(argc, argv, "--llm-gdn-heads", 16);
    uint32_t gated_delta_hidden = parse_uint_arg(argc, argv, "--llm-gdn-hidden", 496);
    uint32_t parameter_seed = parse_uint_arg(argc, argv, "--llm-seed", 42);
    if ((gated_delta_heads != 8u && gated_delta_heads != 16u &&
         gated_delta_heads != 32u && gated_delta_heads != 64u) ||
        gated_delta_hidden == 0u || gated_delta_hidden % 16u != 0u) {
        printf("[llm] GDN heads must be 8, 16, 32, or 64; FFN width must be a positive multiple of 16\n");
        return;
    }
    constexpr float kGatedDeltaLearningRate = 2.0e-4f;
    constexpr float kAttentionLearningRate = 1.0e-4f;
    constexpr float kRopeBase = 10000.0f;
    constexpr uint32_t kValSeeds = 5;
    uint32_t head_dim = kModelDim / gated_delta_heads;
    uint64_t gated_delta_state_bytes = uint64_t(kLayerCount) *
        gated_delta_heads * head_dim * head_dim * sizeof(float16_t);
    uint32_t train_steps = parse_uint_arg(argc, argv, "--llm-steps", 20000);
    std::string checkpoint = parse_string_arg(argc, argv, "--llm-checkpoint", "");
    uint32_t log_interval =
        (std::max)(1u, parse_uint_arg(argc, argv, "--llm-log-interval", 500));
    std::string selected_model = parse_string_arg(argc, argv, "--llm-model", "compare");
    std::filesystem::path output_dir = parse_string_arg(
        argc, argv, "--llm-output", "output");
    bool run_all = selected_model == "compare" || selected_model == "all";
    bool run_attention = run_all || selected_model == "attention";
    bool run_gated_delta = run_all || selected_model == "gated-delta";
    bool run_yoco = selected_model == "yoco-swa";
    bool run_yoco_conv = run_all || selected_model == "yoco" || selected_model == "yoco-conv";
    uint32_t conv_kernel = parse_uint_arg(argc, argv, "--llm-conv-kernel", 3);
    if (run_yoco_conv && (conv_kernel < 1u || conv_kernel > 32u))
        throw std::runtime_error("Convolution kernel must be between 1 and 32 tokens");
    if (!run_attention && !run_gated_delta && !run_yoco && !run_yoco_conv) {
        printf("[llm] unknown --llm-model '%s'; expected attention, "
               "gated-delta, yoco, yoco-conv, yoco-swa, compare, or all\n",
               selected_model.c_str());
        return;
    }

    printf("=== main_llm: softmax attention, Gated DeltaNet, and YOCO (default: Conv3) ===\n");
    printf("config | model %s | steps %u | batch %u | dim %u | sequence %u\n",
           selected_model.c_str(),
           train_steps,
           kBatchSize,
           kModelDim,
           kSeqActiveLen);

    const std::vector<Mesh> base_shapes{
        make_cube_triangles(),
        make_tetrahedron_triangles(),
    };

    std::vector<ValSeed> val(kValSeeds);
    for (uint32_t seed = 0; seed < kValSeeds; ++seed) {
        std::mt19937 validation_rng(9001 + seed);
        std::vector<float> mesh_target;
        generate_batch(base_shapes, kBatchSize, validation_rng, mesh_target);
        build_token_batch(mesh_target,
                          kBatchSize,
                          val[seed].input_tokens,
                          val[seed].target_tokens);
        val[seed].first_target_mesh.assign(
            mesh_target.begin(),
            mesh_target.begin() + kTrianglesPerMesh * kMeshFeatureDim);
    }

    std::vector<float> sample_condition_meshes(
        kBatchSize * kTrianglesPerMesh * kMeshFeatureDim,
        0.0f);
    for (uint32_t batch = 0; batch < kBatchSize; ++batch) {
        const auto& source = val[batch % kValSeeds].first_target_mesh;
        float* destination =
            sample_condition_meshes.data() + batch * kTrianglesPerMesh * kMeshFeatureDim;
        std::copy(source.begin(), source.end(), destination);
    }

    std::filesystem::create_directories(output_dir);
    std::vector<ExperimentResult> results;
    auto run_model = [&](const char* name, AttentionMode mode, uint32_t hidden_dim,
                         uint32_t heads, float learning_rate, uint32_t window = 32u) {
        TokenModel model(kBatchSize, kSeqLen, kVocabSize, kModelDim, hidden_dim,
                         kLayerCount, kRopeBase, mode, heads, window);
        initialize_model(model, parameter_seed, checkpoint);
        results.push_back(train_experiment(model, name, train_steps, log_interval,
            learning_rate, base_shapes, val, sample_condition_meshes, output_dir));
    };
    if (run_attention) {
        run_model("attention", AttentionMode::Softmax, kAttentionHiddenDim,
                  1u, kAttentionLearningRate);
    }

    if (run_gated_delta) {
        printf("GDN config | heads %u | head_dim %u | FFN %u | seed %u\n",
               gated_delta_heads, head_dim, gated_delta_hidden, parameter_seed);
        run_model("gated_delta", AttentionMode::GatedDelta, gated_delta_hidden,
                  gated_delta_heads, kGatedDeltaLearningRate);
    }
    if (run_yoco) {
        run_model("yoco", AttentionMode::Yoco, kYocoHiddenDim, 1u, kAttentionLearningRate);
    }
    if (run_yoco_conv) {
        printf("YOCO causal depthwise convolution | kernel %u | local layers %u\n", conv_kernel, kLayerCount / 2u);
        run_model("yoco_conv", AttentionMode::YocoConv, kYocoHiddenDim, 1u, kAttentionLearningRate, conv_kernel);
    }
    printf("\n=== comparison (lower is better) ===\n");
    std::ofstream comparison_csv(output_dir / "llm_comparison.csv");
    comparison_csv << "model,parameters,best_step,val_ce,completion_mse,"
                      "train_update_ms,validation_ms,validation_runs,validation_mean_ms,"
                      "decode_ms,total_train_seconds,"
                      "inference_state_bytes,state_growth,parameter_bytes,training_tensor_bytes,"
                      "inference_tensor_bytes,cached_logit_rmse,reference_decode_ms,reference_completion_mse\n";
    for (const ExperimentResult& result : results) {
        double update_ms =
            result.train_update_seconds * 1000.0 / double((std::max)(1u, train_steps));
        double validation_mean_ms =
            result.validation_seconds * 1000.0 / double((std::max)(1u, result.validation_runs));
        printf("%-16s | params %llu | best_step %u | val_ce %.6f | completion_mse %.6f | update_ms %.3f | validation_mean_ms %.3f | decode_ms %.3f | state_bytes %llu (%s)\n",
               result.name.c_str(),
               static_cast<unsigned long long>(result.parameters),
               result.best_step,
               result.best.val_ce,
               result.best.val_completion_mse,
               update_ms,
               validation_mean_ms,
               result.decode_ms,
               static_cast<unsigned long long>(result.inference_state_bytes),
               result.inference_state_grows_with_sequence ? "linear" : "fixed");
        comparison_csv << result.name << ","
                       << result.parameters << ","
                       << result.best_step << ","
                       << result.best.val_ce << ","
                       << result.best.val_completion_mse << ","
                       << update_ms << ","
                       << result.validation_seconds * 1000.0 << ","
                       << result.validation_runs << ","
                       << validation_mean_ms << ","
                       << result.decode_ms << ","
                       << result.train_seconds << ","
                       << result.inference_state_bytes << ","
                       << (result.inference_state_grows_with_sequence ? "linear" : "fixed") << ","
                       << result.parameters * sizeof(float16_t) << ","
                       << result.training_tensor_bytes << "," << result.inference_tensor_bytes << ","
                       << result.cached_logit_rmse << "," << result.reference_decode_ms << ","
                       << result.reference_completion_mse
                       << "\n";
    }
    if (run_gated_delta) {
        printf("gated delta inference state | %llu bytes per sequence across %u layers "
               "(fixed with sequence length)\n",
               static_cast<unsigned long long>(gated_delta_state_bytes),
               kLayerCount);
    }
    fflush(stdout);
}
