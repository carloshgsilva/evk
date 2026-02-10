## main_llm.cpp - Unconditional mesh generation via flow matching

This document summarizes what `main_llm.cpp` now does.

### Objective
- Train a small generative model that maps random noise directly to triangle meshes.
- Use flow matching targets to learn a vector field from noise to data.
- Time conditioning is injected through a reserved feature channel.

### Data representation
- A mesh is a fixed list of `kTrianglesPerMesh = 16` triangle slots.
- Each triangle slot uses `kTriangleFeatureDim = 16` features (logical 10 used):
  - 9 values for 3D triangle coordinates (3 vertices x 3 coords)
  - 1 existence flag (`exist`) scaled by `kExistScale = 4.0`
  - remaining padded values are zero (the last padded slot is reused to carry time $t$ during training/sampling)
- All `kTrianglesPerMesh` slots are enabled by default.
- Base cube triangles (`kCubeTriangleCount = 12`) are repeated to fill the 16 slots.

### Batch format
`MeshBatch` contains only:
- `noise`: Gaussian noise tensor (`batch_size x tri_count x kNoiseDim`)
- `target`: full triangle feature tensor (`batch_size x tri_count x feature_dim`)

There is no conditional/partial mesh tensor.

### Model
- Input tensor shape: `[B, T, noise_dim]`
- Output tensor shape: `[B, T, feature_dim]`
- Pipeline:
  - linear projection of noise (`w_in`)
  - positional triangle embedding (`tri_emb`)
  - stack of attention blocks
  - RMSNorm before attention/FFN blocks
  - output projection (`w_out`) back to feature_dim
  - output predicted flow velocity
- Loss: MSE against the flow-matching velocity target.

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

### Run
Use:

```bat
.\run.bat --llm
```

This configures/builds, runs training, and writes OBJ outputs under `output/`.
