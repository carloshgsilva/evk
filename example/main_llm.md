## `main_llm.cpp` - Conditional causal autoregressive mesh completion (cross-entropy)

This demo trains a conditional token autoregressive mesh completion model instead of flow matching.

### Objective
- Condition on the first 2 mesh triangles and autoregressively complete the remaining 10 triangles.
- Train with next-token prediction and cross-entropy.
- Use explicit sequence markers:
  - `BOS` (beginning of sequence)
  - `EOS` (end of sequence)

### Tokenization
Coordinates are quantized into **128 bins**:
- Coordinate range: `[-1.32, 1.32]`
- Tokens:
  - `0` = PAD / IGNORE
  - `1` = BOS
  - `2` = EOS
  - `3..130` = coordinate bins (128 values)

Vocabulary is padded to a tile-friendly size (`144`) for GPU matmul constraints.

### Sequence layout
For each mesh:
- 12 cube triangles in canonical face order
- 9 coordinate values per triangle
- coordinate token count = `12 * 9 = 108`

Sequence (active part):
- `BOS` + 108 coord tokens + `EOS` = 110 tokens

Sequence tensor length is padded to `160` (multiple of 16) for compute kernel alignment.

Conditioning and supervision:
- The first 2 triangles are copied into the input sequence as a fixed prefix (`18` coordinate tokens).
- Cross-entropy is masked over that prefix.
- The model is supervised only on the remaining 10 triangles plus `EOS`.

### Model
- Embedding lookup for token IDs
- Rotary position encoding (RoPE) applied to attention `q` and `k`
- Stack of causal attention blocks (RMSNorm + attention + GELU FFN + residual)
- Output projection to vocabulary logits
- Loss: cross entropy (`target=0` positions ignored)

### Training and metrics
The run prints:
- `train_ce`: training cross-entropy
- `val_ce`: validation cross-entropy over the predicted completion region
- `val_completion_mse`: greedy-decoded coordinate MSE on the 10 predicted triangles only

### Outputs
The demo only exports `output/mesh_val_evolution.obj`. It keeps the full target
mesh in the first column of each row and appends conditioned completion samples
over training so target/prediction comparisons stay meaningful.

### Run
Use:

```bat
.\run.bat --llm
```

This configures/builds the project and runs the LLM demo.
