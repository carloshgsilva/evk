# OIDN weights

The OIDN integration loads the official fast or balanced RT LDR model at
runtime, with either color-only or color+albedo+normal inputs. The binary
weights are intentionally not stored in this repository. The demo defaults to
the fast color-only model.

From the repository root, download them with:

```powershell
New-Item -ItemType Directory -Force oidn-weights | Out-Null
curl.exe -L https://media.githubusercontent.com/media/RenderKit/oidn-weights/master/rt_ldr.tza -o oidn-weights/rt_ldr.tza
curl.exe -L https://media.githubusercontent.com/media/RenderKit/oidn-weights/master/rt_ldr_small.tza -o oidn-weights/rt_ldr_small.tza
curl.exe -L https://media.githubusercontent.com/media/RenderKit/oidn-weights/master/rt_ldr_alb_nrm.tza -o oidn-weights/rt_ldr_alb_nrm.tza
curl.exe -L https://media.githubusercontent.com/media/RenderKit/oidn-weights/master/rt_ldr_alb_nrm_small.tza -o oidn-weights/rt_ldr_alb_nrm_small.tza
```

The weights are distributed by Intel/RenderKit under the Apache 2.0 license.
