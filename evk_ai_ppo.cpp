#include "evk_ai_ppo.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "evk_ai.h"

namespace evk::ai::ppo {

namespace {

constexpr uint32_t kRows = 64; // rows per batch entry of the (envs / 64, 64, width) activations

uint32_t round_up(uint32_t value, uint32_t multiple) { return (value + multiple - 1u) / multiple * multiple; }

// xorshift64*: fast, and the same sequence on every platform.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed * 0x9E3779B97F4A7C15ull + 1u) {}
    uint64_t next() {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return state * 0x2545F4914F6CDD1Dull;
    }
    float uniform() { return float(next() >> 40) * (1.0f / 16777216.0f); }
};

// One layer: out = [relu](in @ w [+ residual]).
struct Stage {
    Tensor* in;
    Tensor* w;
    Tensor* out;
    Tensor* residual = nullptr;
    bool relu = false;
    uint8_t tile = 16;                    // column tile of the forward and input-gradient matmuls
    Tensor* grad = nullptr;               // loss gradient at `out` before its ReLU
    Tensor* in_grad = nullptr;            // where the input's gradient goes; null for the observations
    bool in_grad_started = false;         // that buffer already holds the residual path's share
    std::unique_ptr<Tensor> per_batch;    // weight gradient of each batch entry
};

} // namespace

struct Agent::Impl {
    Config config;
    uint32_t width = 0;       // observation columns: obs_size, the constant 1, padding to 16
    uint32_t head_width = 0;  // action logits and the value, padded to 16
    uint32_t step = 0;        // rollout steps recorded so far
    Rng rng;

    std::vector<std::unique_ptr<Tensor>> tensors; // weights first, then activations
    std::vector<Tensor*> weights;
    std::vector<AdamState> adam;
    std::vector<Stage> stages;
    Tensor* obs = nullptr;
    Tensor* head = nullptr;
    std::unique_ptr<Tensor> rollout_obs, samples;

    // Per rollout slot (step * envs + env); values has one more step for the bootstrap.
    std::vector<uint32_t> actions;
    std::vector<float> log_probs, values, rewards;
    std::vector<uint8_t> dones;
    std::vector<float> episode_returns; // running undiscounted return of each environment
    double entropy_sum = 0.0, finished_return_sum = 0.0;
    uint32_t finished = 0;

    explicit Impl(const Config& c) : config(c), rng(c.seed) {
        if (config.actions < 1 || config.actions > 15) throw std::invalid_argument("ppo: actions must be 1 to 15");
        if (config.obs_size == 0) throw std::invalid_argument("ppo: obs_size must be set");
        if (config.envs == 0 || config.envs % kRows != 0) throw std::invalid_argument("ppo: envs must be a multiple of 64");
        if (config.hidden == 0 || config.hidden % 16 != 0) throw std::invalid_argument("ppo: hidden must be a multiple of 16");
        if (config.horizon == 0) throw std::invalid_argument("ppo: horizon must be at least 1");
        width = round_up(config.obs_size + 1u, 16u);
        head_width = round_up(config.actions + 1u, 16u);
        const uint32_t envs = config.envs, hidden = config.hidden, groups = envs / kRows;

        auto make = [&](Shape shape) -> Tensor* {
            tensors.push_back(std::make_unique<Tensor>(shape));
            return tensors.back().get();
        };
        // Weights in forward order, He initialised. The second weight of a residual block
        // starts small so the block starts close to the identity; the head starts near zero
        // so the first policy is close to uniform.
        srand(config.seed);
        auto weight = [&](uint32_t in, uint32_t out, float deviation) {
            weights.push_back(make({in, out}));
            weights.back()->random_init(deviation);
        };
        const float he = std::sqrt(2.0f / float(hidden));
        weight(width, hidden, std::sqrt(2.0f / float(width)));
        for (uint32_t b = 0; b < std::max(config.blocks, 1u); ++b) {
            weight(hidden, hidden, he);
            if (config.blocks > 0) weight(hidden, hidden, 0.1f * he);
        }
        weight(hidden, head_width, 0.01f);
        adam.resize(weights.size());

        // Layers. Matmul tiles: 64 rows (one batch entry) and the widest column tile that
        // divides both the layer's input and output width.
        const uint8_t wide = hidden % 64u == 0u ? 64 : 16;
        obs = make({groups, kRows, width});
        size_t next = 0;
        auto linear = [&](Tensor* in, uint8_t tile, Tensor* residual) -> Stage& {
            Tensor* w = weights[next++];
            stages.push_back(Stage{in, w, make({groups, kRows, w->shape[1]}), residual});
            stages.back().tile = tile;
            stages.back().relu = true;
            return stages.back();
        };
        Tensor* h = linear(obs, 16, nullptr).out;
        if (config.blocks == 0) {
            h = linear(h, wide, nullptr).out;
        } else {
            for (uint32_t b = 0; b < config.blocks; ++b) {
                Tensor* inner = linear(h, wide, nullptr).out;
                h = linear(inner, wide, h).out;
            }
        }
        Stage& last = linear(h, 16, nullptr);
        last.relu = false;
        head = last.out;

        // Gradient buffers. A stage's gradient is the loss gradient at its output before the
        // ReLU. A residual connection passes that gradient on unchanged, so the stage that
        // produced the residual input shares the buffer: the whole residual stream has one
        // gradient buffer, and the branch adds into it.
        auto producer = [&](Tensor* tensor) -> Stage* {
            for (Stage& stage : stages)
                if (stage.out == tensor) return &stage;
            return nullptr;
        };
        for (size_t i = stages.size(); i-- > 0;) {
            Stage& stage = stages[i];
            if (!stage.grad) stage.grad = &stage.out->grad();
            if (stage.residual) producer(stage.residual)->grad = stage.grad;
        }
        for (Stage& stage : stages) {
            if (Stage* from = producer(stage.in)) {
                stage.in_grad = from->grad;
                stage.in_grad_started = from->grad != &from->out->grad();
            }
            stage.per_batch = std::make_unique<Tensor>(Shape({groups, stage.w->shape[0], stage.w->shape[1]}));
        }

        rollout_obs = std::make_unique<Tensor>(Shape({config.horizon * envs, width}));
        samples = std::make_unique<Tensor>(Shape({config.horizon * envs, 4u}));
        const size_t slots = size_t(config.horizon) * envs;
        actions.resize(slots);
        log_probs.resize(slots);
        rewards.resize(slots);
        dones.resize(slots);
        values.resize(slots + envs);
        episode_returns.assign(envs, 0.0f);
    }

    void forward_stages() {
        for (const Stage& stage : stages) {
            evk::ai::matmul(*stage.in, *stage.w, *stage.out, false, false, false, 64, stage.tile, stage.residual);
            if (stage.relu) evk::ai::relu(*stage.out, *stage.out);
        }
    }

    // Backward pass of forward_stages:
    // - weight gradients are computed per batch entry and then summed, written rather than
    //   accumulated, so nothing has to be zeroed;
    // - a layer's input gradient is written by one matmul (added where the residual stream's
    //   buffer already holds the skip path's share) and then masked in place by the ReLU of
    //   that input;
    // - the observations get no gradient.
    void backward_stages() {
        for (size_t i = stages.size(); i-- > 0;) {
            Stage& stage = stages[i];
            uint32_t in = stage.w->shape[0], out = stage.w->shape[1];
            evk::ai::matmul(*stage.in, *stage.grad, *stage.per_batch, true, false, false,
                            uint8_t(in % 64u == 0u ? 64 : 16), uint8_t(out % 64u == 0u ? 64 : 16));
            evk::ai::sum_batch(*stage.per_batch, stage.w->grad(), config.envs / kRows, in * out, false);
            if (!stage.in_grad) continue;
            evk::ai::matmul(*stage.grad, *stage.w, *stage.in_grad, false, true, stage.in_grad_started, 64, stage.tile);
            evk::ai::relu_backward(*stage.in_grad, *stage.in, *stage.in_grad, false);
        }
    }

    // Runs the network on host observations; head->cpu() is valid afterwards. With
    // rollout_slice >= 0 the observations are also kept on the GPU as that rollout step.
    void forward(const float* data, int rollout_slice) {
        float16_t* staged = obs->cpu();
        const uint32_t n = config.obs_size;
        for (uint32_t e = 0; e < config.envs; ++e) {
            float16_t* row = staged + size_t(e) * width;
            for (uint32_t i = 0; i < n; ++i) row[i] = float16_t(data[size_t(e) * n + i]);
            row[n] = float16_t(1.0f); // stands in for the bias terms
            for (uint32_t i = n + 1; i < width; ++i) row[i] = float16_t(0.0f);
        }
        obs->cpu_upload(false);
        auto& cmd = evk::ai::GetCmd();
        if (rollout_slice >= 0) {
            const size_t bytes = size_t(config.envs) * width * sizeof(float16_t);
            cmd.copy(obs->buffer, rollout_obs->buffer, bytes, 0, size_t(rollout_slice) * bytes);
            cmd.barrier();
        }
        forward_stages();
        head->cpu_download(false);
        evk::ai::SubmitCmd(true);
    }

    // Softmax of one head row. Returns false for non-finite logits.
    bool probabilities(const float16_t* row, float* p) const {
        float top = -1e30f;
        for (uint32_t a = 0; a < config.actions; ++a) {
            p[a] = float(row[a]);
            if (!std::isfinite(p[a])) return false;
            top = std::max(top, p[a]);
        }
        float sum = 0.0f;
        for (uint32_t a = 0; a < config.actions; ++a) { p[a] = std::exp(p[a] - top); sum += p[a]; }
        for (uint32_t a = 0; a < config.actions; ++a) p[a] /= sum;
        return true;
    }
};

Agent::Agent(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Agent::~Agent() = default;

const Config& Agent::config() const { return impl_->config; }

void Agent::act(const float* obs, uint32_t* actions) {
    Impl& s = *impl_;
    if (s.step >= s.config.horizon) throw std::logic_error("ppo: act() past the horizon; call update()");
    s.forward(obs, int(s.step));
    const float16_t* head = s.head->cpu();
    const uint32_t A = s.config.actions;
    float p[16];
    for (uint32_t e = 0; e < s.config.envs; ++e) {
        const float16_t* row = head + size_t(e) * s.head_width;
        size_t slot = size_t(s.step) * s.config.envs + e;
        uint32_t action = 0;
        float log_prob = 0.0f;
        if (s.probabilities(row, p)) {
            float u = s.rng.uniform(), entropy = 0.0f;
            action = A - 1u; // when rounding leaves u above the sum
            for (uint32_t a = 0; a < A; ++a) {
                if (u < p[a]) { action = a; break; }
                u -= p[a];
            }
            for (uint32_t a = 0; a < A; ++a) entropy -= p[a] * std::log(std::max(p[a], 1e-8f));
            s.entropy_sum += entropy;
            log_prob = std::log(std::max(p[action], 1e-8f));
        } else {
            // A broken network: act uniformly so the rollout stays usable.
            action = uint32_t(s.rng.next() % A);
            log_prob = -std::log(float(A));
        }
        float value = float(row[A]);
        s.actions[slot] = action;
        s.log_probs[slot] = log_prob;
        s.values[slot] = std::isfinite(value) ? value : 0.0f;
        actions[e] = action;
    }
}

void Agent::reward(const float* rewards, const uint8_t* dones) {
    Impl& s = *impl_;
    for (uint32_t e = 0; e < s.config.envs; ++e) {
        size_t slot = size_t(s.step) * s.config.envs + e;
        s.rewards[slot] = rewards[e];
        s.dones[slot] = dones[e] ? 1 : 0;
        s.episode_returns[e] += rewards[e];
        if (dones[e]) {
            s.finished_return_sum += s.episode_returns[e];
            ++s.finished;
            s.episode_returns[e] = 0.0f;
        }
    }
    ++s.step;
}

Stats Agent::update(const float* next_obs) {
    Impl& s = *impl_;
    const uint32_t T = s.config.horizon, N = s.config.envs, A = s.config.actions;
    if (s.step != T) throw std::logic_error("ppo: update() needs horizon act()/reward() steps");

    // Value of where the rollout ends, then GAE.
    s.forward(next_obs, -1);
    const float16_t* head = s.head->cpu();
    for (uint32_t e = 0; e < N; ++e) {
        float value = float(head[size_t(e) * s.head_width + A]);
        s.values[size_t(T) * N + e] = std::isfinite(value) ? value : 0.0f;
    }
    const size_t slots = size_t(T) * N;
    std::vector<float> advantages(slots), returns(slots);
    for (uint32_t e = 0; e < N; ++e) {
        float gae = 0.0f;
        for (int t = int(T) - 1; t >= 0; --t) {
            size_t slot = size_t(t) * N + e;
            float alive = s.dones[slot] ? 0.0f : 1.0f;
            float delta = s.rewards[slot] + s.config.gamma * s.values[slot + N] * alive - s.values[slot];
            gae = delta + s.config.gamma * s.config.lambda * alive * gae;
            advantages[slot] = gae;
            returns[slot] = gae + s.values[slot];
        }
    }

    // Normalised advantages into the samples, (action, old log-probability, advantage, return).
    double mean = 0.0, variance = 0.0, value_loss = 0.0, reward_sum = 0.0;
    for (size_t slot = 0; slot < slots; ++slot) mean += advantages[slot];
    mean /= double(slots);
    for (size_t slot = 0; slot < slots; ++slot) variance += (advantages[slot] - mean) * (advantages[slot] - mean);
    const float adv_mean = float(mean), adv_scale = 1.0f / (float(std::sqrt(variance / double(slots))) + 1e-6f);
    float16_t* sample = s.samples->cpu();
    for (size_t slot = 0; slot < slots; ++slot) {
        sample[slot * 4 + 0] = float16_t(float(s.actions[slot]));
        sample[slot * 4 + 1] = float16_t(s.log_probs[slot]);
        sample[slot * 4 + 2] = float16_t((advantages[slot] - adv_mean) * adv_scale);
        sample[slot * 4 + 3] = float16_t(returns[slot]);
        double error = double(returns[slot]) - double(s.values[slot]);
        value_loss += 0.5 * error * error;
        reward_sum += s.rewards[slot];
    }
    s.samples->cpu_upload(false);

    // Every minibatch is one rollout step, in a fresh random order each epoch. The loss runs
    // on the GPU, so all epochs go out in one submission.
    std::vector<uint32_t> order;
    for (uint32_t epoch = 0; epoch < s.config.epochs; ++epoch) {
        size_t first = order.size();
        for (uint32_t t = 0; t < T; ++t) order.push_back(t);
        for (size_t i = T - 1; i > 0; --i) std::swap(order[first + i], order[first + s.rng.next() % (i + 1)]);
    }
    const size_t slice_bytes = size_t(N) * s.width * sizeof(float16_t);
    for (uint32_t slice : order) {
        auto& cmd = evk::ai::GetCmd();
        cmd.barrier();
        cmd.copy(s.rollout_obs->buffer, s.obs->buffer, slice_bytes, size_t(slice) * slice_bytes);
        cmd.barrier();
        s.forward_stages();
        evk::ai::ppo_loss(*s.head, *s.samples, slice * N, s.head->grad(), A, s.config.clip, s.config.entropy,
                          s.config.value_coef, s.config.grad_scale);
        s.backward_stages();
        for (size_t i = 0; i < s.weights.size(); ++i)
            evk::ai::adam(*s.weights[i], s.weights[i]->grad(), s.adam[i], s.config.lr, 0.9f, 0.999f, 1e-5f,
                          s.config.stochastic_rounding);
    }
    evk::ai::SubmitCmd(true);

    Stats stats;
    stats.episodes = s.finished;
    stats.episode_return = s.finished ? float(s.finished_return_sum / s.finished) : 0.0f;
    stats.step_reward = float(reward_sum / double(slots));
    stats.entropy = float(s.entropy_sum / double(slots));
    stats.value_loss = float(value_loss / double(slots));
    s.step = 0;
    s.finished = 0;
    s.finished_return_sum = 0.0;
    s.entropy_sum = 0.0;
    return stats;
}

void Agent::evaluate(const float* obs, float* probabilities, float* values) {
    Impl& s = *impl_;
    s.forward(obs, -1);
    const float16_t* head = s.head->cpu();
    const uint32_t A = s.config.actions;
    for (uint32_t e = 0; e < s.config.envs; ++e) {
        const float16_t* row = head + size_t(e) * s.head_width;
        if (probabilities && !s.probabilities(row, probabilities + size_t(e) * A))
            for (uint32_t a = 0; a < A; ++a) probabilities[size_t(e) * A + a] = 1.0f / float(A);
        if (values) values[e] = float(row[A]);
    }
}

void Agent::save(const std::string& path) {
    for (Tensor* w : impl_->weights) w->cpu_download(false);
    evk::ai::SubmitCmd(true);
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("ppo: cannot write " + path);
    for (Tensor* w : impl_->weights) file.write((const char*)w->cpu(), w->shape.count() * sizeof(float16_t));
}

void Agent::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("ppo: cannot open " + path);
    for (Tensor* w : impl_->weights) {
        file.read((char*)w->cpu(), w->shape.count() * sizeof(float16_t));
        if (!file) throw std::runtime_error("ppo: " + path + " does not match the network shape");
        w->cpu_upload(false);
    }
    evk::ai::SubmitCmd(true);
}

} // namespace evk::ai::ppo
