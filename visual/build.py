"""Preprocess the Mojito-Light EMRI bricks into compact products for the explorer page.

Usage: python build.py [--sources 0 3 5] [--force]
Also provides track(), the FEW trajectory -> frequency-track helper used by serve.py.
"""
import argparse
import glob
import json
import os
import re
from functools import lru_cache

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import spectrogram
from scipy.signal.windows import tukey
from mojito import MojitoL1File

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOJITO = '/Users/bertd/Documents/PhD/LISA/data/MojitoLight/brickmarket/mojito_light_v1_0_0/data'
DATA_DIR = f'{MOJITO}/EMRI/L1_0p4Hz'
NOISE_FILE = f'{MOJITO}/INSTRUMENT/L1_0p4Hz/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5'
CATALOG = os.path.join(ROOT, 'mojito_catalog_standardized_processed.hdf5')
CHAINS_DIR = os.path.join(ROOT, 'output')
OUT = os.path.join(HERE, 'data')

# Trajectory t=0 (catalog TimeReferenceSSBFrame) lies T_OFF seconds before the first L1 sample.
# The 0.4 Hz bricks are downsampled from the 4 Hz originals: 1000 x 0.25 s + 550 s + 50.5 s = 850.5 s
# (= t0_L1 - TimeReferenceSSBFrame of the official catalog emri_cat_mojito_lite_processed_MT.hdf5).
T_OFF = 850.5
F_MIN = 1e-5                # lower edge of the SNR band (as in PE_validation.py)
HARMONICS = ((2, 0, 0), (2, 0, 1))
LABELS = ['M/M_\\odot', '\\mu/M_\\odot', 'a', 'p_0', 'e_0', 'D_L/\\mathrm{Gpc}',
          '\\theta_S', '\\phi_S', '\\theta_K', '\\phi_K', '\\Phi_{\\phi,0}', '\\Phi_{r,0}']


# ---------------------------------------------------------------- FEW trajectory (only FEW code allowed)
@lru_cache(maxsize=1)
def _traj():
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode import KerrEccEqFlux
    return EMRIInspiral(func=KerrEccEqFlux)


def track(M, mu, a, p0, e0, x0, T_obs, harmonics=HARMONICS, t_off=T_OFF, npts=2000):
    """Frequency tracks f_mkn = |m f_phi + n f_r| [Hz] vs time since the first L1 sample [s].

    T_obs is the data span in seconds; the trajectory is integrated over T_obs + t_off and the part
    before the first L1 sample (t < t_off) is dropped. SSB-frame frequencies, no LISA Doppler.
    """
    from few.utils.geodesic import get_fundamental_frequencies
    from few.utils.constants import MTSUN_SI, YRSID_SI
    x0 = float(np.rint(x0))
    t, p, e, x = _traj()(M, mu, a, p0, e0, x0, T=(T_obs + t_off) / YRSID_SI, dt=10.0)[:4]
    tt = np.linspace(t_off, t[-1], npts) if t[-1] > t_off else np.array([])
    pp, ee = CubicSpline(t, p)(tt), CubicSpline(t, e)(tt)
    om_phi, _, om_r = get_fundamental_frequencies(a, pp, ee, np.full_like(pp, x0))
    f_phi, f_r = om_phi / (2 * np.pi * M * MTSUN_SI), om_r / (2 * np.pi * M * MTSUN_SI)
    return dict(t_data=tt - t_off, p=pp, e=ee, f_phi=f_phi, f_r=f_r,
                tracks={f'{m}{k}{n}': np.abs(m * f_phi + n * f_r) for m, k, n in harmonics},
                t_plunge=float(t[-1]), e_final=float(e[-1]), p_final=float(p[-1]),
                plunged=bool(t[-1] < T_obs + t_off - 10.0))


def tolist(d):
    """Recursively convert numpy containers to JSON-safe python (NaN/inf -> None)."""
    if isinstance(d, dict):
        return {k: tolist(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [tolist(v) for v in d]
    if isinstance(d, np.ndarray):
        return tolist(d.tolist())
    if isinstance(d, (np.floating, float)):
        return float(d) if np.isfinite(d) else None
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    return d


def dump(obj, path):
    with open(path, 'w') as f:
        json.dump(tolist(obj), f, separators=(',', ':'), allow_nan=False)


# ---------------------------------------------------------------- inputs
def find_brick(k, data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, f'EMRI_731d_2.5s_L1_source{k}_*.h5')))
    return files[-1] if files else None


def load_catalog(k):
    with h5py.File(CATALOG, 'r') as f:
        return {key: float(ds[k]) for key, ds in f['Binaries'].items()}


def load_noise(noise_file):
    """Day-averaged one-sided noise covariance [1/Hz] for XYZ and AET on the log-f grid."""
    with MojitoL1File(noise_file) as f:
        nu0 = f.laser_frequency
        fn = f.noise_estimates.freq_sampling.f()
        ts = f.noise_estimates.time_sampling
        S = {c: np.mean(getattr(f.noise_estimates, c)[:], axis=0) / nu0 ** 2 for c in ('xyz', 'aet')}
    return fn, S, (ts.t0, ts.t0 + ts.duration)


def ref_snrs():
    txt = open(os.path.join(CHAINS_DIR, 'results.md')).read()
    return {int(k): float(v) for k, v in re.findall(r'Source (\d+) summary.*?SNR data:\s+([\d.]+)', txt, re.S)}


def truth_vector(cat):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ecl = SkyCoord(ra=cat['RightAscension'] * u.rad, dec=cat['Declination'] * u.rad,
                   frame='icrs').barycentrictrueecliptic
    return [cat['PrimaryMassSSBFrame'], cat['SecondaryMassSSBFrame'], cat['PrimarySpinParameter'],
            cat['SemiLatusRectum'], cat['Eccentricity'], cat['LuminosityDistance'] * 1e-3,
            np.pi / 2 - ecl.lat.rad, ecl.lon.rad, cat['PolarAnglePrimarySpin'],
            cat['AzimuthalAnglePrimarySpin'], cat['AzimuthalPhase'], cat['RadialPhase']]


# ---------------------------------------------------------------- frequency domain
def inner(a_fft, b_fft, Cinv, df):
    """<a|b> = 4 df Re sum_k a^dagger C^-1 b, for a_fft, b_fft of shape (n_freq, 3) and Cinv (n_freq, 3, 3)."""
    return 4 * df * np.real(np.einsum('fj,fjk,fk->', np.conj(a_fft), Cinv, b_fft))


def cov_on(fn, S, f):
    """Cubic-spline (in log f) interpolation of the 3x3 covariance onto f."""
    lf, out = np.log(fn), np.empty((len(f), 3, 3), complex)
    for i in range(3):
        for j in range(3):
            out[:, i, j] = CubicSpline(lf, S[:, i, j].real)(np.log(f)) + 1j * CubicSpline(lf, S[:, i, j].imag)(np.log(f))
    return out


def snr(ffts, f, df, fn, S, chunk=1 << 20):
    """Network SNR with the full 3x3 covariance and single-channel SNRs, streamed in frequency chunks."""
    rho2, rho2_c = 0.0, np.zeros(3)
    for s in range(0, len(f), chunk):
        sl = slice(s, s + chunk)
        C = cov_on(fn, S, f[sl])
        d = ffts[:, sl].T
        rho2 += inner(d, d, np.linalg.inv(C), df)
        rho2_c += 4 * df * np.sum(np.abs(d) ** 2 / C[:, range(3), range(3)].real, axis=0)
    return np.sqrt(rho2), np.sqrt(rho2_c)


def log_bin(f, power, nbins=3000):
    """Mean of power in geometric frequency bins; returns bin centres and means (empty bins dropped)."""
    edges = np.geomspace(f[0], f[-1] * (1 + 1e-12), nbins + 1)
    idx = np.searchsorted(f, edges)
    cs = np.concatenate([[0.0], np.cumsum(power)])
    n = np.diff(idx)
    ok = n > 0
    return np.sqrt(edges[:-1] * edges[1:])[ok], ((cs[idx[1:]] - cs[idx[:-1]])[ok] / n[ok])


def rebin_rows(f, P, edges):
    """Average spectrogram rows (power density) into log-f bins; empty bins take the nearest row."""
    idx = np.searchsorted(f, edges)
    cs = np.vstack([np.zeros(P.shape[1]), np.cumsum(P, axis=0)])
    n = np.diff(idx)
    out = (cs[idx[1:]] - cs[idx[:-1]]) / np.maximum(n, 1)[:, None]
    centres = np.sqrt(edges[:-1] * edges[1:])
    empty = n == 0
    out[empty] = P[np.clip(np.searchsorted(f, centres[empty]), 0, len(f) - 1)]
    return out


def envelope(x, nbins=4000):
    m = -(-len(x) // nbins)
    xp = np.concatenate([x, np.full(m * nbins - len(x), x[-1])]).reshape(nbins, m)
    return xp.min(axis=1), xp.max(axis=1)


# ---------------------------------------------------------------- per-source build
def build_source(k, args, noise):
    out = os.path.join(OUT, f'source_{k}')
    if os.path.exists(os.path.join(out, 'params.json')) and not args.force:
        print(f'source {k}: exists, skipping (use --force)')
        return json.load(open(os.path.join(out, 'params.json')))
    brick = find_brick(k, args.data_dir)
    if brick is None:
        print(f'source {k}: no brick found')
        return None
    os.makedirs(out, exist_ok=True)
    fn, S, noise_span = noise
    cat = load_catalog(k)
    print(f'source {k}: {os.path.basename(brick)}')

    with MojitoL1File(brick) as f:
        ts, nu0 = f.tdis.time_sampling, f.laser_frequency
        dt, N, t0 = ts.dt, ts.size, ts.t0
        T_obs = N * dt
        freqs = np.fft.rfftfreq(N, d=dt)
        band = freqs >= F_MIN
        fb, df = freqs[band], 1.0 / (N * dt)
        w = tukey(N, alpha=0.01)
        sedges = np.geomspace(1e-4, 0.2, 401)
        td, fd, spec, ffts, parseval = {}, {'f': None}, [], {}, []
        for grp, chans in (('XYZ', 'XYZ'), ('AET', 'AET')):
            ffts[grp] = np.empty((3, band.sum()), complex)
            for i, c in enumerate(chans):
                d = getattr(f.tdis, f'{c.lower()}2')[:] / nu0
                if grp == 'XYZ':
                    td[c] = envelope(d)
                    fs_, ts_, P = spectrogram(d, fs=1 / dt, window='hann', nperseg=2 ** 15, noverlap=2 ** 14,
                                              scaling='density', return_onesided=True)
                    spec.append(rebin_rows(fs_, P, sedges))
                    del P
                X = dt * np.fft.rfft(w * d)
                if grp == 'XYZ':
                    pw = np.abs(X) ** 2
                    lhs = dt * np.sum((w * d) ** 2)
                    rhs = df * (pw[0] + 2 * pw[1:-1].sum() + pw[-1] * (1 if N % 2 == 0 else 2))
                    parseval.append(abs(lhs - rhs) / lhs)
                    fbin, pbin = log_bin(fb, pw[band])
                    fd['f'] = fbin
                    fd[c] = 2 * np.sqrt(fbin * pbin)
                ffts[grp][i] = X[band]
                del d, X
            print(f'  {grp} channels done')

    rho, rho_c = snr(ffts['XYZ'], fb, df, fn, S['xyz'])
    rho_aet, rho_aet_c = snr(ffts['AET'], fb, df, fn, S['aet'])
    del ffts
    for i, c in enumerate('XYZ'):  # log-log interpolation of the PSD for plotting (stays positive near TDI nulls)
        fd[f'n{c}'] = np.sqrt(fd['f'] * np.exp(np.interp(np.log(fd['f']), np.log(fn), np.log(S['xyz'][:, i, i].real))))
    dump(fd, os.path.join(out, 'fd.json'))
    dump({'t_days': (np.arange(4000) + 0.5) * (-(-N // 4000)) * dt / 86400,
          **{c: {'min': v[0].astype(np.float32), 'max': v[1].astype(np.float32)} for c, v in td.items()}},
         os.path.join(out, 'td_overview.json'))

    # spectrogram: one-sided PSD [1/Hz], log10, floored 20 decades below the max
    spec = np.array(spec)
    spec = np.log10(np.maximum(spec, spec.max() * 1e-20)).astype('<f4')
    spec.tofile(os.path.join(out, 'spec.bin'))
    t_cols = ts_ / 86400
    dump({'f': np.sqrt(sedges[:-1] * sedges[1:]), 'f_edges': sedges, 't_days': t_cols,
          'shape': spec.shape, 'channels': ['X', 'Y', 'Z'], 'units': 'log10 one-sided PSD [1/Hz] (fractional frequency)'},
         os.path.join(out, 'spec.json'))

    # trajectories: data span (true track) and 10 yr (plunge diagnostics)
    pars = (cat['PrimaryMassSSBFrame'], cat['SecondaryMassSSBFrame'], cat['PrimarySpinParameter'],
            cat['SemiLatusRectum'], cat['Eccentricity'], np.cos(cat['InclinationAngle']))
    tr = track(*pars, T_obs)
    dump(tr, os.path.join(out, 'track_true.json'))
    from few.utils.constants import YRSID_SI
    long = track(*pars, 10 * YRSID_SI, npts=500)

    # track consistency: spectrogram ridge (X) vs the nearest harmonic |m f_phi + n f_r| (m=1..4, n=-2..30),
    # in log-f bin units; for eccentric sources the dominant harmonic is not f_200
    dlog = np.log(sedges[1] / sedges[0])
    cols = (t_cols * 86400 > tr['t_data'][0]) & (t_cols * 86400 < tr['t_data'][-1]) if len(tr['t_data']) else []
    ridge_off, ridge_harm, ridge_off_200 = None, None, None
    if np.any(cols):
        ridge = sedges[:-1][np.argmax(spec[0][:, cols], axis=0)] * np.exp(dlog / 2)
        tc = t_cols[cols] * 86400
        fp, fr = (np.interp(tc, tr['t_data'], tr[q]) for q in ('f_phi', 'f_r'))
        mn = [(m, n) for m in range(1, 5) for n in range(-2, 31)]
        off = np.abs(np.log(ridge[None] / np.abs(np.array([m * fp + n * fr for m, n in mn]))))
        best = np.argmin(off, axis=0)
        ridge_off = float(np.median(off[best, range(len(tc))]) / dlog)
        m, n = mn[np.bincount(best).argmax()]
        ridge_harm = f'{m}0{n}'
        ridge_off_200 = float(np.median(np.abs(np.log(ridge / (2 * np.abs(fp))))) / dlog)

    # chains
    chain_file = os.path.join(args.chains_dir, f'source_{k}', f'PE_run_source_{k}.h5')
    has_chains = os.path.exists(chain_file)
    truth = truth_vector(cat)
    if has_chains:
        with h5py.File(chain_file, 'r') as h:
            n_it = int(h['mcmc'].attrs['iteration'])  # datasets are preallocated; only n_it rows are filled
            ch = h['mcmc/chain/model_0'][:n_it, 0, :, 0, :].astype('<f4')
            ll = h['mcmc/log_like'][:n_it, 0, :].astype('<f4')
            acc = h['mcmc/moves/StretchMove_0/acceptance_fraction'][0]
        with open(os.path.join(out, 'chains.bin'), 'wb') as fh:
            fh.write(ch.tobytes())
            fh.write(ll.tobytes())
        try:
            from eryn.backends import HDFBackend
            tau_eryn = HDFBackend(chain_file, read_only=True).get_autocorr_time(discard=ch.shape[0] // 2)['model_0'][0]
        except Exception as exc:  # cross-check only
            print(f'  eryn autocorr failed: {exc}')
            tau_eryn = None
        dump({'N_it': ch.shape[0], 'N_w': ch.shape[1], 'N_p': ch.shape[2], 'labels': LABELS, 'truth': truth,
              'acceptance_fraction': acc, 'tau_eryn': tau_eryn}, os.path.join(out, 'chains.json'))

    in_band = (fn >= F_MIN) & (fn <= 0.5 / dt)
    Sd = S['xyz'][in_band][:, range(3), range(3)].real
    ref = ref_snrs().get(k)
    params = {
        'k': k, 'brick': os.path.basename(brick), 'has_chains': has_chains,
        'catalog': cat, 'labels': LABELS, 'truth': truth,
        'meta': {'t0': t0, 'dt': dt, 'N': N, 'T_obs': T_obs, 'nu0': nu0, 't_off': T_OFF},
        'derived': {
            'T_plunge': long['t_plunge'], 'T_plunge_data': long['t_plunge'] - T_OFF, 'plunges_in_data': long['t_plunge'] < T_obs + T_OFF,
            'e_final': long['e_final'], 'p_final': long['p_final'],
            'f_phi_final': abs(long['f_phi'][-1]) if len(long['f_phi']) else None,
            'e_end_data': tr['e'][-1] if len(tr['e']) else None, 'p_end_data': tr['p'][-1] if len(tr['p']) else None,
        },
        'snr': {'XYZ': rho, 'AET': rho_aet, 'X': rho_c[0], 'Y': rho_c[1], 'Z': rho_c[2],
                'A': rho_aet_c[0], 'E': rho_aet_c[1], 'T': rho_aet_c[2], 'reference': ref},
        'checks': {
            'snr_rel_err': abs(rho - ref) / ref if ref else None,
            'parseval_max_rel_err': max(parseval),
            'ridge_offset_bins': ridge_off, 'ridge_harmonic': ridge_harm, 'ridge_offset_200_bins': ridge_off_200,
            'noise_finite_positive': bool(np.all(np.isfinite(Sd)) and np.all(Sd > 0)),
            'noise_covers_data': bool(noise_span[0] <= t0 + 1e-6 and noise_span[1] >= t0 + T_obs - 1e-6),
        },
    }
    dump(params, os.path.join(out, 'params.json'))
    print(f'  SNR_XYZ = {rho:.3f} (reference {ref}), SNR_AET = {rho_aet:.3f}, ridge offset = {ridge_off} bins (dominant {ridge_harm})')
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources', type=int, nargs='+', default=list(range(8)))
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--noise-file', default=NOISE_FILE)
    ap.add_argument('--chains-dir', default=CHAINS_DIR)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    noise = load_noise(args.noise_file)
    for k in args.sources:
        build_source(k, args, noise)
    manifest = []
    for k in range(8):
        p = os.path.join(OUT, f'source_{k}', 'params.json')
        if os.path.exists(p):
            q = json.load(open(p))
            c = q['catalog']
            manifest.append({'k': k, 'brick': q['brick'], 'has_chains': q['has_chains'], 'SNR': q['snr']['XYZ'],
                             'M': c['PrimaryMassSSBFrame'], 'mu': c['SecondaryMassSSBFrame'],
                             'a': c['PrimarySpinParameter'], 'e0': c['Eccentricity']})
    dump(manifest, os.path.join(OUT, 'manifest.json'))


if __name__ == '__main__':
    main()
