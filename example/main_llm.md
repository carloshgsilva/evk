# Mesh language-model comparison

`main_llm.cpp` compares two eight-layer autoregressive models on procedural
mesh completion:

- causal softmax Transformer attention, the quality baseline;
- Gated DeltaNet, the constant-memory recurrent alternative.

Both models use a 256-wide token representation, RoPE, residual MLP blocks,
FP16 parameters, activations, optimizer state, and inference state. FP32 is used
only for temporary accumulation inside compute shaders.

## Dataset and objective

Each example is a randomly rotated and translated cube or subdivided
tetrahedron. The first two triangles are supplied as conditioning tokens and the
model autoregressively completes the remaining ten triangles. Training uses
next-token cross entropy. Validation reports cross entropy and completion MSE
over generated, non-conditioning coordinates.

Only `*_mesh_val_evolution.obj` is produced for mesh visualization. Each
validation checkpoint appends five generated meshes to the file.

## Softmax attention

The baseline uses eight standard causal self-attention blocks with RoPE and a
512-wide feed-forward path. It has 4,268,032 trainable parameters. Its equivalent
FP16 K/V cache grows linearly with sequence length.

## Gated DeltaNet

The Gated DeltaNet also uses eight residual blocks and exactly 4,268,032
trainable parameters. Each layer has 64 four-dimensional heads and a 448-wide
feed-forward path. The recurrent update is:

```text
S'[t] = alpha[t] * S[t - 1]
e[t]  = v[t] - k[t]^T * S'[t]
S[t]  = S'[t] + beta[t] * k[t] * e[t]^T
o[t]  = q[t]^T * S[t]
```

Q and K are L2-normalized and receive RoPE before the state update. Across all
eight layers, recurrent state occupies 16 KiB of FP16 storage per sequence and
does not grow with context length.

Training uses an exact reverse state-gradient scan to supply FP16 boundaries
for parallel 16-token backward chunks, including the key-dependent delta
correction. Decode uses the same recurrence, with FP16 state storage.

## Running

```bat
.\run.bat --llm
.\run.bat --llm --llm-model attention
.\run.bat --llm --llm-model gated-delta
.\run.bat --llm --llm-model compare
.\run.bat --llm --llm-model compare --llm-steps 20000 --llm-log-interval 250
```

`compare` and `all` run both retained models. The default budget is 20,000
updates.

`--llm-gdn-heads` accepts 8, 16, 32, or 64
(head dimensions 32, 16, 8, or 4). `--llm-gdn-hidden` sets the FFN width,
a positive multiple of 16. Defaults remain 64 heads and FFN width 448.
`--llm-seed` sets parameter initialization (default 42), not the data stream.
`--llm-output` selects a separate output directory; files there are overwritten.
For example, this tests dimension 16 at the baseline parameter count:

```bat
.\run.bat --llm --llm-model gated-delta --llm-gdn-heads 16 --llm-gdn-hidden 496 --llm-output output/gdn16
```

Outputs are written under `output/` by default:

- `attention_training_curve.csv`
- `attention_mesh_val_evolution.obj`
- `gated_delta_training_curve.csv`
- `gated_delta_mesh_val_evolution.obj`
- `llm_comparison.csv`
