## `main_llm.cpp` — Causal autoregressive mesh generation (cross-entropy)

This demo now trains a **token autoregressive model** for meshes instead of flow matching.

### Objective
- Learn mesh generation with a causal transformer-style stack.
- Train with next-token prediction and cross-entropy.
- Use explicit sequence markers:
  - `BOS` (beginning of sequence)
  - `EOS` (end of sequence)

### Tokenization
Coordinates are quantized into **128 bins**:
- Coordinate range: `[-1.25, 1.25]`
- Tokens:
  - `0` = PAD / IGNORE
  - `1` = BOS
  - `2` = EOS
  - `3..130` = coordinate bins (128 values)

Vocabulary is padded to a tile-friendly size (`144`) for GPU matmul constraints.

### Sequence layout
For each mesh:
- 16 triangle slots
- 9 coordinate values per triangle
- coordinate token count = `16 * 9 = 144`

Sequence (active part):
- `BOS` + 144 coord tokens + `EOS` = 146 tokens

Sequence tensor length is padded to `160` (multiple of 16) for compute kernel alignment.

### Model
- Embedding lookup for token IDs
- Rotary position encoding (RoPE) applied to attention `q` and `k`
- Stack of causal attention blocks (RMSNorm + attention + FFN + residual)
- Output projection to vocabulary logits
- Loss: cross entropy (`target=0` positions ignored)

### Training and metrics
The run prints:
- `train_ce`: training cross-entropy
- `val_ce`: validation cross-entropy
- `tf_err`: teacher-forced token error rate
- `noise_rate`: autoregressive run-to-run token jitter (instability)
- `noise_mse`: autoregressive run-to-run coordinate jitter MSE
- `fit_mse`: decoded autoregressive coordinate MSE vs validation targets

`noise_rate` and `noise_mse` are expected to be near zero with greedy decoding.

### Outputs
After training, the demo exports:
- `output/mesh_target.obj`
- `output/mesh_pred.obj`
- `output/mesh_pred_seed0.obj`
- `output/mesh_pred_seed1.obj`
- `output/mesh_pred_seed2.obj`

### Run
Use:

```bat
.\run.bat --llm
```

This configures/builds the project and runs the LLM demo.
