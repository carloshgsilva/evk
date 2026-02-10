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
#include <thread>

#include <evk_ai.h>

namespace {
constexpr uint32_t kCubeTriangleCount = 12;
constexpr uint32_t kTrianglesPerMesh = 16; // padded to satisfy matmul tile constraints
constexpr uint32_t kTriangleFeatureDim = 16; // padded feature dim
constexpr uint32_t kUsedFeatureDim = 10; // 9 coords + exist
constexpr uint32_t kNoiseDim = 16;
constexpr uint32_t kTimeChannel = kTriangleFeatureDim - 1;
constexpr float kExistScale = 4.0f;
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
    std::vector<float> noise;
    std::vector<float> target;
};

void generate_noise(uint32_t batch_size, float noise_scale, std::mt19937& rng, MeshBatch& batch) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    uint32_t count = batch_size * kTrianglesPerMesh * kNoiseDim;
    batch.noise.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        batch.noise[i] = dist(rng) * noise_scale;
    }

    // Zero out padded feature channels (including the time channel).
    if (kNoiseDim == kTriangleFeatureDim) {
        uint32_t sample_dim = kTrianglesPerMesh * kNoiseDim;
        for (uint32_t b = 0; b < batch_size; ++b) {
            uint32_t base = b * sample_dim;
            for (uint32_t t = 0; t < kTrianglesPerMesh; ++t) {
                uint32_t offset = base + t * kNoiseDim;
                for (uint32_t d = kUsedFeatureDim; d < kNoiseDim; ++d) {
                    batch.noise[offset + d] = 0.0f;
                }
            }
        }
    }
}

void generate_batch(const std::vector<std::array<float, 9>>& base,
                    uint32_t batch_size,
                    std::mt19937& rng,
                    MeshBatch& batch) {
    uint32_t sample_dim = kTrianglesPerMesh * kTriangleFeatureDim;
    batch.target.assign(batch_size * sample_dim, 0.0f);

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
            uint32_t base_idx = (b * kTrianglesPerMesh + t) * kTriangleFeatureDim;

            for (uint32_t i = 0; i < 9; ++i) {
                batch.target[base_idx + i] = tri[i];
            }
            batch.target[base_idx + 9] = kExistScale;
        }
    }
}

void upload_tensor(Tensor& t, const std::vector<float>& data) {
    assert(t.shape.count() == data.size());
    float16_t* ptr = t.cpu();
    for (uint32_t i = 0; i < t.shape.count(); ++i) {
        ptr[i] = float16_t(data[i]);
    }
    t.cpu_upload();
}

void download_tensor(Tensor& t, std::vector<float>& data) {
    t.cpu_download();
    float16_t* ptr = t.cpu();
    data.resize(t.shape.count());
    for (uint32_t i = 0; i < t.shape.count(); ++i) {
        data[i] = float(ptr[i]);
    }
}

void build_flow_matching_targets(const std::vector<float>& x0,
                                 const std::vector<float>& x1,
                                 uint32_t batch_size,
                                 uint32_t tri_count,
                                 uint32_t feature_dim,
                                 std::mt19937& rng,
                                 std::vector<float>& x_t,
                                 std::vector<float>& v_target,
                                 std::vector<float>& t_batch) {
    assert(x0.size() == x1.size());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    uint32_t sample_dim = tri_count * feature_dim;

    x_t.resize(x0.size());
    v_target.resize(x0.size());
    t_batch.resize(batch_size);

    for (uint32_t b = 0; b < batch_size; ++b) {
        float t = dist(rng);
        t_batch[b] = t;
        uint32_t base = b * sample_dim;
        for (uint32_t i = 0; i < sample_dim; ++i) {
            uint32_t channel = i % feature_dim;
            float x0v = x0[base + i];
            float x1v = x1[base + i];
            if (channel == kTimeChannel) {
                x_t[base + i] = t;
                v_target[base + i] = 0.0f;
            } else {
                x_t[base + i] = (1.0f - t) * x0v + t * x1v;
                v_target[base + i] = x1v - x0v;
            }
        }
    }
}

void inject_time_channel(std::vector<float>& data,
                         uint32_t batch_size,
                         uint32_t tri_count,
                         uint32_t feature_dim,
                         float t) {
    uint32_t sample_dim = tri_count * feature_dim;
    for (uint32_t b = 0; b < batch_size; ++b) {
        uint32_t base = b * sample_dim;
        for (uint32_t tri = 0; tri < tri_count; ++tri) {
            data[base + tri * feature_dim + kTimeChannel] = t;
        }
    }
}

float compute_mse(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(sum / double(a.size()));
}

float compute_rms(const std::vector<float>& a) {
    double sum = 0.0;
    for (float v : a) {
        sum += double(v) * double(v);
    }
    return float(std::sqrt(sum / double(a.size())));
}

float compute_pairwise_seed_mse(const std::vector<std::vector<float>>& seed_preds) {
    if (seed_preds.size() < 2) {
        return 0.0f;
    }
    const size_t sample_dim = seed_preds[0].size();
    double total_mse = 0.0;
    uint32_t pairs = 0;
    for (size_t i = 0; i < seed_preds.size(); ++i) {
        for (size_t j = i + 1; j < seed_preds.size(); ++j) {
            assert(seed_preds[j].size() == sample_dim);
            double mse = 0.0;
            for (size_t d = 0; d < sample_dim; ++d) {
                double diff = double(seed_preds[i][d]) - double(seed_preds[j][d]);
                mse += diff * diff;
            }
            total_mse += mse / double(sample_dim);
            pairs++;
        }
    }
    return pairs ? float(total_mse / double(pairs)) : 0.0f;
}

void compute_exist_stats(const std::vector<float>& pred,
                         uint32_t batch_size,
                         uint32_t tri_count,
                         uint32_t feature_dim,
                         float exist_scale,
                         float& mean_exist) {
    double sum_exist = 0.0;
    uint32_t count = 0;

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t t = 0; t < tri_count; ++t) {
            uint32_t base = (b * tri_count + t) * feature_dim;
            float pred_exist = pred[base + 9] / exist_scale;
            sum_exist += pred_exist;
            count++;
        }
    }

    mean_exist = count ? float(sum_exist / double(count)) : 0.0f;
}

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
        uint32_t base_idx = t * kTriangleFeatureDim;
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

// Append a single-sample mesh to an OBJ file, applying an X/Y offset and
// updating the running vertex_index so faces remain correct across appends.
void append_obj(const std::filesystem::path& path,
                const std::vector<float>& features,
                uint32_t tri_count,
                uint32_t& vertex_index,
                float x_offset = 0.0f,
                float y_offset = 0.0f,
                float exist_threshold = 0.5f) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out;
    for (int attempt = 0; attempt < 8; ++attempt) {
        out.open(path, std::ios::app);
        if (out.is_open()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!out.is_open()) {
        printf("warning: failed to append evolution mesh to %s\n", path.string().c_str());
        return;
    }

    for (uint32_t t = 0; t < tri_count; ++t) {
        uint32_t base_idx = t * kTriangleFeatureDim;
        float exist = features[base_idx + 9] / kExistScale;
        // Passing a very low threshold disables exist culling entirely.
        if (exist_threshold > -1.0e8f && exist < exist_threshold) {
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

    out.close();
}

Tensor& cross_attention(Graph& graph, Tensor& q, Tensor& k, Tensor& v, float scale = 0.0f) {
    assert(q.shape.rank() == 3 && k.shape.rank() == 3 && v.shape.rank() == 3);
    uint32_t B = q.shape[0];
    uint32_t Nq = q.shape[1];
    uint32_t Nk = k.shape[1];
    uint32_t D = q.shape[2];
    assert(k.shape[2] == D && v.shape[2] == D);

    Tensor& scores = graph.tensor({B, Nq, Nk});
    Tensor& probs = graph.tensor({B, Nq, Nk});
    Tensor& out = graph.tensor({B, Nq, D});

    float attn_scale = (scale > 0.0f) ? scale : (1.0f / std::sqrt(float(D)));

    out.forward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale]() {
        evk::ai::matmul(q, k, scores, false, true, false, 16, 16);
        evk::ai::scale(scores, attn_scale);
        evk::ai::softmax(scores, probs);
        evk::ai::matmul(probs, v, out, false, false, false, 16, 16);
    };

    out.backward_fn = [&q, &k, &v, &scores, &probs, &out, attn_scale]() {
        evk::ai::matmul(out.grad(), v, probs.grad(), false, true, false, 16, 16);
        evk::ai::matmul(probs, out.grad(), v.grad(), true, false, true, 16, 16);
        evk::ai::softmax_backward(probs, probs.grad(), scores.grad(), 1.0f);
        evk::ai::scale(scores.grad(), attn_scale);
        evk::ai::matmul(scores.grad(), k, q.grad(), false, false, true, 16, 16);
        evk::ai::matmul(scores.grad(), q, k.grad(), true, false, true, 16, 16);
    };

    return out;
}

struct CrossAttentionBlock {
    Tensor* w_q = nullptr;
    Tensor* w_k = nullptr;
    Tensor* w_v = nullptr;
    Tensor* w_o = nullptr;
    Tensor* w1 = nullptr;
    Tensor* w2 = nullptr;

    uint32_t embed_dim = 0;
    uint32_t hidden_dim = 0;

    void init(Graph& graph, uint32_t embed_dim_, uint32_t hidden_dim_) {
        embed_dim = embed_dim_;
        hidden_dim = hidden_dim_;

        w_q = &graph.tensor({embed_dim, embed_dim}, true);
        w_k = &graph.tensor({embed_dim, embed_dim}, true);
        w_v = &graph.tensor({embed_dim, embed_dim}, true);
        w_o = &graph.tensor({embed_dim, embed_dim}, true);
        w1 = &graph.tensor({embed_dim, hidden_dim}, true);
        w2 = &graph.tensor({hidden_dim, embed_dim}, true);
    }

    void init_weights(float base_scale) {
        float attn_scale = base_scale * (1.0f / sqrtf(float(embed_dim)));
        float ffn_scale = base_scale * sqrtf(2.0f / float(embed_dim));
        float out_scale = base_scale * sqrtf(2.0f / float(hidden_dim));

        w_q->random_init(attn_scale);
        w_k->random_init(attn_scale);
        w_v->random_init(attn_scale);
        w_o->random_init(attn_scale);
        w1->random_init(ffn_scale);
        w2->random_init(out_scale);
    }

    Tensor& forward(Graph& graph, Tensor& input, float residual_scale = 1.0f) {
        Tensor& norm_in = graph.rms_norm(input);
        Tensor& q = graph.matmul(norm_in, *w_q);
        Tensor& k = graph.matmul(norm_in, *w_k);
        Tensor& v = graph.matmul(norm_in, *w_v);
        Tensor& attn_out = cross_attention(graph, q, k, v);
        Tensor& attn_proj = graph.matmul(attn_out, *w_o);

        Tensor& attn_residual = graph.add(input, attn_proj);

        Tensor& norm_ffn = graph.rms_norm(attn_residual);
        Tensor& hidden = graph.matmul(norm_ffn, *w1);
        Tensor& hidden_relu = graph.relu(hidden);
        Tensor& hidden_proj = graph.matmul(hidden_relu, *w2);

        Tensor& output = graph.add(attn_residual, hidden_proj);
        return output;
    }
};

struct MeshCompletionModel {
    uint32_t batch_size;
    uint32_t tri_count;
    uint32_t feature_dim;
    uint32_t noise_dim;
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t num_layers;

    Graph graph;
    Tensor* input = nullptr;
    Tensor* target = nullptr;
    Tensor* pred = nullptr;
    Tensor* loss = nullptr;

    Tensor* w_in = nullptr;
    Tensor* tri_emb = nullptr;
    Tensor* w_out = nullptr;
    std::vector<CrossAttentionBlock> blocks;

    MeshCompletionModel(uint32_t batch_size_, uint32_t tri_count_, uint32_t feature_dim_,
                        uint32_t noise_dim_, uint32_t embed_dim_, uint32_t hidden_dim_,
                        uint32_t num_layers_)
        : batch_size(batch_size_),
          tri_count(tri_count_),
          feature_dim(feature_dim_),
          noise_dim(noise_dim_),
          embed_dim(embed_dim_),
          hidden_dim(hidden_dim_),
          num_layers(num_layers_) {
        build_graph();
    }

    void build_graph() {
        input = &graph.tensor({batch_size, tri_count, noise_dim});
        target = &graph.tensor({batch_size, tri_count, feature_dim});

        w_in = &graph.tensor({noise_dim, embed_dim}, true);
        tri_emb = &graph.tensor({tri_count, embed_dim}, true);
        w_out = &graph.tensor({embed_dim, feature_dim}, true);

        blocks.resize(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            blocks[i].init(graph, embed_dim, hidden_dim);
        }

        Tensor& input_emb = graph.matmul(*input, *w_in);
        Tensor& x0 = graph.add_position_embedding(input_emb, *tri_emb, batch_size, tri_count);

        Tensor* x = &x0;
        float residual_scale = 1.0f;
        for (uint32_t i = 0; i < num_layers; ++i) {
            x = &blocks[i].forward(graph, *x, residual_scale);
        }

        Tensor& out = graph.matmul(*x, *w_out);
        pred = &out;
        loss = &graph.mse_loss(*pred, *target);
    }

    void init_weights(uint32_t seed = 42) {
        srand(seed);
        float base_scale = 0.2f;
        w_in->random_init(base_scale / sqrtf(float(noise_dim)));
        tri_emb->random_init(0.02f);

        for (auto& block : blocks) {
            block.init_weights(base_scale);
        }
        float out_scale = base_scale * sqrtf(2.0f / float(embed_dim));
        w_out->random_init(out_scale);
    }
};

void flow_match_sample(MeshCompletionModel& model,
                       const std::vector<float>& x0,
                       uint32_t steps,
                       std::vector<float>& x_out) {
    if (steps == 0) {
        x_out = x0;
        return;
    }

    x_out = x0;
    std::vector<float> vel;
    std::vector<float> dummy_target(x_out.size(), 0.0f);
    upload_tensor(*model.target, dummy_target);

    float dt = 1.0f / float(steps);
    for (uint32_t s = 0; s < steps; ++s) {
        float t = (float(s) + 0.5f) * dt;
        inject_time_channel(x_out, model.batch_size, model.tri_count, model.feature_dim, t);

        upload_tensor(*model.input, x_out);
        model.graph.eval(false);
        download_tensor(*model.pred, vel);

        for (size_t i = 0; i < x_out.size(); ++i) {
            if ((i % model.feature_dim) == kTimeChannel) {
                x_out[i] = t;
                continue;
            }
            x_out[i] += dt * vel[i];
        }
    }

    inject_time_channel(x_out, model.batch_size, model.tri_count, model.feature_dim, 0.0f);
}

} // namespace

void main_llm() {
    evk::ai::initialize();
    printf("=== main_llm: unconditional mesh generation via flow matching ===\n");

    auto start = std::chrono::high_resolution_clock::now();

    constexpr uint32_t kBatchSize = 64;
    constexpr uint32_t kEmbedDim = 32;
    constexpr uint32_t kHiddenDim = 128;
    constexpr uint32_t kLayers = 4;
    constexpr uint32_t kTrainSteps = 10000;
    constexpr uint32_t kSampleSteps = 10;
    constexpr float kLearningRate = 1.0e-3f;
    constexpr float kNoiseScale = 1.0f;

    MeshCompletionModel model(kBatchSize, kTrianglesPerMesh, kTriangleFeatureDim,
                              kNoiseDim, kEmbedDim, kHiddenDim, kLayers);
    model.init_weights(42);

    std::mt19937 train_rng(1337);
    auto base = make_cube_triangles();

    uint32_t sample_dim = kTrianglesPerMesh * kTriangleFeatureDim;

    // Create multiple validation seeds so we can visualize several
    // generated meshes in separate rows of the evolution OBJ.
    constexpr uint32_t kValSeeds = 3;
    std::vector<MeshBatch> val_batches(kValSeeds);
    std::vector<std::vector<float>> val_target_singles(kValSeeds, std::vector<float>(sample_dim));

    for (uint32_t s = 0; s < kValSeeds; ++s) {
        std::mt19937 rng_val(9001 + s);
        generate_batch(base, kBatchSize, rng_val, val_batches[s]);
        generate_noise(kBatchSize, kNoiseScale, rng_val, val_batches[s]);
        for (uint32_t i = 0; i < sample_dim; ++i) {
            val_target_singles[s][i] = val_batches[s].target[i];
        }
    }

    // Create an OBJ that will collect the validation prediction evolution.
    // We'll write the target (left) and then append successive validation
    // predictions to the right so they don't overlap.
    std::filesystem::path evo_path("output/mesh_val_evolution.obj");
    if (std::filesystem::exists(evo_path)) {
        std::filesystem::remove(evo_path);
    }
    uint32_t evo_vertex_index = 1;
    uint32_t evo_snapshot_count = 0;
    const float mesh_spacing = 2.0f; // offset applied between successive meshes
    const float row_spacing = mesh_spacing * 2.0f; // vertical spacing between seed rows
    // Keep all triangles in evolution snapshots so rows don't disappear when
    // predicted existence logits are temporarily below threshold.
    const float evo_exist_threshold = -1.0e9f;

    // append target (left) for each validation seed
    for (uint32_t s = 0; s < kValSeeds; ++s) {
        float y_offset = -float(s) * row_spacing;
        append_obj(evo_path, val_target_singles[s], kTrianglesPerMesh, evo_vertex_index,
                   0.0f, y_offset, evo_exist_threshold);
    }

    std::vector<float> pred_velocity;
    std::vector<float> flow_x;
    std::vector<float> flow_v_target;
    std::vector<float> flow_t_batch;
    std::vector<float> pred_mesh;

    uint32_t log_interval = (kTrainSteps <= 20) ? 1u : 200u;

    for (uint32_t step = 1; step <= kTrainSteps; ++step) {
        MeshBatch batch;
        generate_batch(base, kBatchSize, train_rng, batch);
        generate_noise(kBatchSize, kNoiseScale, train_rng, batch);

        build_flow_matching_targets(batch.noise, batch.target,
                                    kBatchSize, kTrianglesPerMesh, kTriangleFeatureDim,
                                    train_rng, flow_x, flow_v_target, flow_t_batch);

        upload_tensor(*model.input, flow_x);
        upload_tensor(*model.target, flow_v_target);

        model.graph.eval(true);
        model.graph.step_adam(kLearningRate, 0.9f, 0.98f, 1e-4f);
        evk::ai::SubmitCmd(true);

        download_tensor(*model.pred, pred_velocity);
        float train_loss = float(model.loss->item());

        if (step % log_interval == 0 || step == 1 || step == kTrainSteps) {
            float velocity_mse = compute_mse(pred_velocity, flow_v_target);
            float val_mse = 0.0f;
            std::vector<std::vector<float>> seed_preds(kValSeeds, std::vector<float>(sample_dim));

            // append prediction snapshot for each validation seed in its own row
            for (uint32_t s = 0; s < kValSeeds; ++s) {
                flow_match_sample(model, val_batches[s].noise, kSampleSteps, pred_mesh);
                if (s == 0) {
                    val_mse = compute_mse(pred_mesh, val_batches[s].target);
                }

                std::vector<float> val_pred_single(sample_dim);
                for (uint32_t i = 0; i < sample_dim; ++i) {
                    val_pred_single[i] = pred_mesh[i];
                }
                seed_preds[s] = val_pred_single;
                float y_offset = -float(s) * row_spacing;
                append_obj(evo_path, val_pred_single, kTrianglesPerMesh, evo_vertex_index,
                           mesh_spacing * float(evo_snapshot_count + 1), y_offset,
                           evo_exist_threshold);
            }
            float seed_div_mse = compute_pairwise_seed_mse(seed_preds);
            printf("step %4u | flow_loss %.6f | vel_mse %.6f | val_mse %.6f | seed_div_mse %.6f\n",
                   step, train_loss, velocity_mse, val_mse, seed_div_mse);
            fflush(stdout);
            evo_snapshot_count++;
        }
    }

    // evaluate metrics on the first validation seed
    flow_match_sample(model, val_batches[0].noise, kSampleSteps, pred_mesh);
    // save seed-0 prediction for export
    std::vector<float> pred_seed0 = pred_mesh;

    float final_val_mse = compute_mse(pred_mesh, val_batches[0].target);
    float exist_mean = 0.0f;
    compute_exist_stats(pred_mesh, kBatchSize, kTrianglesPerMesh,
                        kTriangleFeatureDim, kExistScale, exist_mean);
    printf("validation mse: %.6f | exist mean: %.3f\n",
        final_val_mse, exist_mean);

    std::vector<float> target_features(sample_dim);
    for (uint32_t i = 0; i < sample_dim; ++i) {
        target_features[i] = val_batches[0].target[i];
    }

    std::vector<std::vector<float>> final_seed_preds(kValSeeds, std::vector<float>(sample_dim));
    // append the final validation prediction snapshot for each seed (rows)
    for (uint32_t s = 0; s < kValSeeds; ++s) {
        flow_match_sample(model, val_batches[s].noise, kSampleSteps, pred_mesh);
        std::vector<float> final_pred_single(sample_dim);
        for (uint32_t i = 0; i < sample_dim; ++i) {
            final_pred_single[i] = pred_mesh[i];
        }
        final_seed_preds[s] = final_pred_single;
        float y_offset = -float(s) * row_spacing;
        append_obj(evo_path, final_pred_single, kTrianglesPerMesh, evo_vertex_index,
                   mesh_spacing * float(evo_snapshot_count + 1), y_offset,
                   evo_exist_threshold);
        export_obj(std::filesystem::path("output") / ("mesh_pred_seed" + std::to_string(s) + ".obj"),
                   final_pred_single, kTrianglesPerMesh);
    }
    float final_seed_div_mse = compute_pairwise_seed_mse(final_seed_preds);
    printf("final seed_div_mse: %.6f\n", final_seed_div_mse);
    evo_snapshot_count++;

    export_obj("output/mesh_target.obj", target_features, kTrianglesPerMesh);
    export_obj("output/mesh_pred.obj", pred_seed0, kTrianglesPerMesh);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("main_llm() took %.4f seconds\n", duration.count());
    evk::ai::shutdown();
}
