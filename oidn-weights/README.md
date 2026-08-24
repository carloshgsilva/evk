# OIDN weights

The OIDN integration loads the official balanced color-only RT LDR model at
runtime. The binary weights are intentionally not stored in this repository.

From the repository root, download them with:

```powershell
New-Item -ItemType Directory -Force oidn-weights | Out-Null
curl.exe -L https://raw.githubusercontent.com/RenderKit/oidn-weights/master/rt_ldr.tza -o oidn-weights/rt_ldr.tza
```

The weights are distributed by Intel/RenderKit under the Apache 2.0 license.
