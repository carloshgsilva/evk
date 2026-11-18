// PPO (proximal policy optimization, Schulman et al. 2017) for discrete actions on evk::ai.
//
// The agent owns the policy network, the rollout storage and the update; the caller owns
// the environments. Each iteration:
//
//     for (uint32_t t = 0; t < config.horizon; ++t) {
//         agent.act(obs, actions);           // envs x obs_size observations in, one action each out
//         ...step every environment...
//         agent.reward(rewards, dones);      // what those actions earned, and which episodes ended
//     }
//     agent.update(obs);                     // obs after the last step; trains on the rollout
//
// An environment whose episode ended resets itself; the next act() sees its new observation.
//
// Network: obs -> hidden -> hidden -> head with ReLU and no bias terms (the agent appends a
// constant 1 to every observation instead). The head holds the action logits and the state
// value. With blocks > 0 the second hidden layer is replaced by `blocks` residual blocks
// h = relu(h + W2 relu(W1 h)), W2 starting small so a block starts near the identity.
// Every layer is one matmul with the residual add fused in and the ReLU applied in place;
// the update runs all epochs in one GPU submission (evk::ai::ppo_loss computes the loss
// gradient on the GPU).
//
// Weights are FP16 and the update uses Adam with stochastic rounding (see evk::ai::adam).
// Requires evk::ai::initialize().
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace evk::ai::ppo {

struct Config {
    uint32_t obs_size = 0;       // observation values per environment
    uint32_t actions = 0;        // discrete actions, 1 to 15
    uint32_t envs = 4096;        // environments stepped together, a multiple of 64; also the minibatch
    uint32_t horizon = 32;       // steps per environment between updates; each is one minibatch
    uint32_t epochs = 3;         // passes over the rollout per update
    uint32_t hidden = 256;       // hidden width, a multiple of 16
    uint32_t blocks = 0;         // residual blocks; 0 keeps the plain two-hidden-layer network
    float lr = 3e-4f;
    float gamma = 0.995f;        // discount
    float lambda = 0.95f;        // GAE
    float clip = 0.2f;           // surrogate clip range
    float entropy = 0.01f;       // entropy bonus
    float value_coef = 0.5f;     // value loss weight
    // Gradients are per row times this instead of the mean over rows, which keeps FP16
    // gradients out of the subnormal range; Adam is scale invariant apart from epsilon.
    float grad_scale = 1.0f / 64.0f;
    // The weights are around 0.1, where an FP16 step is close to the Adam update; without
    // stochastic rounding most updates are dropped and learning stalls.
    bool stochastic_rounding = true;
    uint32_t seed = 1;           // weights and action sampling
};

struct Stats {
    uint32_t episodes = 0;       // episodes that ended in this rollout
    float episode_return = 0.0f; // their mean undiscounted return
    float step_reward = 0.0f;    // mean reward per step
    float entropy = 0.0f;        // mean policy entropy over the rollout
    float value_loss = 0.0f;     // mean 0.5 (value - return)^2 before the update
};

class Agent {
public:
    explicit Agent(const Config& config);
    ~Agent();
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    const Config& config() const;

    // Samples an action for every environment from obs (envs x obs_size) and records the
    // step. Call it horizon times, each followed by reward(), then update().
    void act(const float* obs, uint32_t* actions);
    // rewards and dones (envs each) of the actions from the last act().
    void reward(const float* rewards, const uint8_t* dones);
    // Trains on the rollout. next_obs (envs x obs_size) is the observation after the last
    // step; its value estimates the return beyond the rollout.
    Stats update(const float* next_obs);

    // Runs the policy without recording anything: probabilities (envs x actions) and values
    // (envs); either may be null.
    void evaluate(const float* obs, float* probabilities, float* values);

    // Raw weights in forward order. load() needs a network of the same shape.
    void save(const std::string& path);
    void load(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace evk::ai::ppo
