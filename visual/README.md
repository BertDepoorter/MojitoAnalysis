# Mojito EMRI Explorer

Interactive page for the 8 Mojito-Light EMRI bricks. It shows the source parameters with validation
checks, the TDI time series, the frequency-domain signal against the noise, a spectrogram with FEW
trajectory tracks, and MCMC chain diagnostics.

```bash
conda activate few_env
python fetch_data.py      # once; needs .env with LISA_CONSORTIUM_NAME / LISA_CONSORTIUM_KEY
python build.py           # once; ~1 min per source; writes visual/data/ (add --force to rebuild, --sources 0 3 to select)
python serve.py           # opens http://127.0.0.1:8765
```

Opening `index.html` directly (`file://`) does not work. The page needs `serve.py` for data fetching,
live FEW trajectories (`/api/trajectory`) and zoomed time-domain windows (`/api/td`).

Conventions:
- Noise: one-sided Mojito `noise_estimates`, averaged over days, from the instrument-noise brick.
- FFT: $\tilde d=\Delta t\,\mathrm{rfft}(w\,d)$ with a Tukey window, α = 0.01. This is the same inner product as `PE_validation.py`.
- Spectrogram: the one-sided signal PSD, not whitened.
- Tracks: $f_{mkn}=|m f_\phi+n f_r|$, computed with the FEW trajectory module only. The defaults are 200 and 201.
  Trajectory $t=0$ is the catalog reference time, 850.5 s before the first L1 sample (bricks downsampled from 4 Hz). SSB frame, no LISA Doppler.
- Chains: the coldest temperature only, truncated to the filled iterations. Thinning is by $\lceil\tau_{\max}\rceil$ after burn-in.
