# Legacy

Superseded scripts and notebooks, kept for reference only. Nothing here is part
of the current analysis; nothing in the repository imports or references it.

## Notebooks

- `FLR_sim_puhti.ipynb` — the Puhti-cluster version of the response comparison.
  Superseded by `../FLR_sim_vsc.ipynb`, which carries the same analysis with the
  later fixes.
- `FLR_EMRI_source_6.ipynb`, `FLR_EMRI_source_7.ipynb` — per-source ancestors of
  the comparison notebook, before it was generalised over source index.
- `FLR_sine_comparison.ipynb` — early sine-wave sanity check of the
  `fastlisaresponse` / Mojito comparison.

## Scripts

- `mojito_emri.py` — script version of the response comparison that loops over
  all sources and writes a summary table. Its timing/orbits helpers
  (`get_mojito_timing`, `create_orbits`) also live in `../timing.ipynb`.

## Container definitions

Three earlier iterations of the Apptainer image, all superseded by
`../lisa_inference.def` (which pins commits of GPUBackendTools,
LISAanalysistools and fastlisaresponse and targets sm_70):

- `lisa_env.def` — first working build, system CUDA, no architecture pinning.
- `lisa_env_gpu.def` — same build against the Conda CUDA toolkit.
- `lisa_envGPU.def` — adds `CMAKE_CUDA_ARCHITECTURES` so the GPU kernels build on
  a node without a GPU.

- `Dockerfile` / `emri_env.yaml` — the Docker route, before the move to
  Apptainer. The `Dockerfile` expects `src/lisasim/EMRI/emri_env.yaml`, a path
  from the `lisasim` repository that does not exist here.
