## main_llm.cpp - Unconditional mesh generation via flow matching

This document summarizes what `main_llm.cpp` now does.

### Objective
- Train a small generative model that maps random noise directly to triangle meshes.
- Use flow matching targets to learn a vector field from noise to data.
- Time conditioning is injected through a reserved feature channel.

### Data representation
- A mesh is a fixed list of `kTrianglesPerMesh = 16` triangle slots.
- Each triangle slot uses `kEmbedDim` channels (logical mesh channels: 10 used):
  - 9 values for 3D triangle coordinates (3 vertices x 3 coords)
  - 1 existence flag (`exist`) scaled by `kExistScale`
  - remaining channels are available latent channels (the final channel is reserved for time $t$ during training/sampling)
- All `kTrianglesPerMesh` slots are enabled by default.
- Base cube triangles (`kCubeTriangleCount`) are repeated to fill the 16 slots.

### Batch format
`MeshBatch` contains only:
- `noise`: Gaussian noise tensor (`batch_size x tri_count x kEmbedDim`)
- `target`: full triangle feature tensor (`batch_size x tri_count x kEmbedDim`)

There is no conditional/partial mesh tensor.

### Model
- Input tensor shape: `[B, T, kEmbedDim]`
- Output tensor shape: `[B, T, kEmbedDim]`
- Pipeline:
  - positional triangle embedding (`tri_emb`)
  - stack of attention blocks
  - RMSNorm before attention/FFN blocks
  - output predicted flow velocity as a residual delta head: `pred = x - input`
- Loss: MSE against the flow-matching velocity target.

The model uses a single shared embedding width end-to-end (input/noise/features/output),
so residual paths do not pass through separate input/output projection layers.

### Stability notes from debugging
- Non-mesh channels (`[kUsedFeatureDim, kEmbedDim)`) in the initial noise are explicitly zeroed,
  including the reserved time channel. This avoids random latent drift from unconstrained channels.
- The velocity head uses `x - input` to prevent identity leakage through deep residual stacks
  after removing `w_out`.

### Flow matching training logic
For each training step:
1. Generate target meshes $x_1$ and noise $x_0$.
2. Sample $t \sim U(0,1)$ per mesh.
3. Build $x_t = (1 - t) x_0 + t x_1$ and inject $t$ into the reserved feature slot.
4. Target velocity is $v = x_1 - x_0$ (time slot target is $0$).
5. Train with MSE between predicted velocity and $v$.

### Validation and outputs
- Logs include:
  - `flow_loss`
  - `vel_mse`
  - `val_mse`
  - `seed_div_mse`
- Final report includes:
  - `validation mse`
  - mean predicted `exist` over all triangle slots
- OBJ outputs:
  - `output/mesh_target.obj`
  - `output/mesh_pred.obj`
  - `output/mesh_val_evolution.obj` (target + prediction snapshots over time)

Sampling uses a short Euler rollout (default 10 steps) from noise to data.
The denoising rollout uses a power-biased $t$ schedule (controlled by
`kSampleSchedulePower`) to spend more steps near $t = 1$ for improved
final precision.

### Run
Use:

```bat
.\run.bat --llm
```

This configures/builds, runs training, and writes OBJ outputs under `output/`.
