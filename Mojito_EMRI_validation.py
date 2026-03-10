# Script to perform some level 3 validation checks for the EMRIs and L2D template. 
# based on FLR_sim.ipynb notebook. Code has been refactored by Claude. 
# All GenAI suggestions have been checked and accepted by Bert Depoorter
source_index = 6

# imports
import os
import glob
import time
import warnings
from copy import deepcopy

import h5py
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal.windows import tukey

# Force backend before importing FLR stack
os.environ["GPUBACKENDTOOLS_FORCE_BACKEND"] = "cuda12x"

import few
import lisatools
import fastlisaresponse
import gpubackendtools

from few.waveform import GenerateEMRIWaveform
from lisaconstants import ASTRONOMICAL_YEAR
from lisaorbits import OEMOrbits
from lisatools.detector import Orbits
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.utils.parallelbase import ParallelModuleBase

from mojito import MojitoL1File
from mojito.download import get_source_params


# ---------------------------
# Configuration
# ---------------------------
YRSID_SI = ASTRONOMICAL_YEAR

SCRATCH = "/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/mojito_light_v1_0_0/data/EMRI/L1"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_INDICES = list(range(8))

# Prefer env vars for credentials
MOJITO_USERNAME = os.getenv("MOJITO_USERNAME", "bert-depoorter")
MOJITO_TOKEN = os.getenv("MOJITO_TOKEN", "ZxUfm^CM97tXUPT")

USE_GPU = True
FORCE_BACKEND = "cuda12x" if USE_GPU else None

# Timing/trim parameters
t_dltt_orbits = 10.0
dt_orbits = 5e5
t_strain_offset = 550.0
n_trim_lolipops = 1000
n_orbit_buffer = 10
window_jaxgbresponse = 0.5
oemorbits_name = "esa-trailing"

# FEW setup
sum_kwargs = {"pad_output": True}
inspiral_kwargs = {"DENSE_STEPPING": 0, 
                   "max_init_len": int(1e8)}
amplitude_kwargs = {}
waveform_kwargs_template = {"mode_selection_threshold": 0.0}
mode_selection_threshold = 0.0

# TDI setup
index_beta = 7
index_lambda = 8
tdi_kwargs_esa_template = {
    "order": 40,
    "tdi": TDIConfig("2nd generation"),
    "tdi_chan": "XYZ",
}

# Noise file
noise_file = (
    f"{SCRATCH}/../../NOISE/L1/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
)


print(
    f"""
few:              {few.__version__}
lisatools:        {lisatools.__version__}
fastlisaresponse: {fastlisaresponse.__version__}
gpubackendtools:  {gpubackendtools.__version__}
"""
)


# ---------------------------
# Utilities
# ---------------------------
def save_fig_to_pdf(pdf, fig):
    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_mojito_timing(
    oemorbits,
    dt,
    t_dltt_orbits,
    dt_orbits,
    t_strain_offset,
    n_trim_lolipops,
    n_orbit_buffer,
    window_jaxgbresponse,
):
    duration_mojito_light = np.ceil(2.0 * ASTRONOMICAL_YEAR / dt) * dt
    orbits = OEMOrbits.from_included(oemorbits)
    t0_orbits = float(orbits.t_start) + t_dltt_orbits

    size_l1 = int(np.round(duration_mojito_light / dt)) + 1
    size_l0 = size_l1 + 2 * n_trim_lolipops
    size_strain = size_l0 + int(2 * np.ceil(t_strain_offset / dt))
    size_orbits = (
        int(np.ceil(((1 + 2 * window_jaxgbresponse) * size_l0 - 1) * dt / dt_orbits))
        + 1
        + 2 * n_orbit_buffer
    )

    t0_l0 = t0_orbits + n_orbit_buffer * dt_orbits + window_jaxgbresponse * size_l0 * dt
    t_init = t0_l0 - t_strain_offset

    return {
        "dt": dt,
        "t_dltt_orbits": t_dltt_orbits,
        "dt_orbits": dt_orbits,
        "t_strain_offset": t_strain_offset,
        "n_trim_lolipops": n_trim_lolipops,
        "n_orbit_buffer": n_orbit_buffer,
        "window_jaxgbresponse": window_jaxgbresponse,
        "t0_orbits": t0_orbits,
        "size_l1": size_l1,
        "size_l0": size_l0,
        "size_strain": size_strain,
        "size_orbits": size_orbits,
        "t0_l0": t0_l0,
        "t_init": t_init,
    }


def create_orbits(oemorbits, timing, dense_orbits_path):
    orbits = OEMOrbits.from_included(oemorbits)
    orbits.write(
        dense_orbits_path,
        dt=timing["dt_orbits"],
        size=timing["size_orbits"],
        t0=timing["t0_orbits"],
        mode="w",
    )


def icrs_to_ecliptic(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    icrs_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    ecliptic_coord = icrs_coord.barycentrictrueecliptic
    return ecliptic_coord.lon.rad, ecliptic_coord.lat.rad


class EMRIWaveBase(ParallelModuleBase):
    def __init__(
        self,
        force_backend=None,
        use_gpu=True,
        inspiral_kwargs=None,
        sum_kwargs=None,
        amplitude_kwargs=None,
        mode_selection_threshold=1e-5,
        t_init=0.0,
        t0_orbits=0.0,
    ):
        super().__init__(force_backend=force_backend)
        self.use_gpu = use_gpu
        self.mode_threshold = mode_selection_threshold
        self.waveform_gen = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            return_list=False,
            inspiral_kwargs=inspiral_kwargs or {},
            sum_kwargs=sum_kwargs or {},
            amplitude_kwargs=amplitude_kwargs or {},
            frame="detector",
        )
        self.t_init = t_init
        self.t0_orbits = t0_orbits

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + b for b in cls.GPU_RECOMMENDED()]

    def __call__(self, *params, T=2.0, dt=5.0):
        wkwargs = deepcopy(waveform_kwargs_template)
        wkwargs["T"] = T
        wkwargs["dt"] = dt
        wkwargs["mode_selection_threshold"] = self.mode_threshold
        return self.waveform_gen(*params, **wkwargs)


def stabilize_covariance_for_inversion(
    cov,
    psd_floor=1e-45,
    smooth_window_logfreq=15,
    return_inverse=True,
):
    cov = np.asarray(cov, dtype=np.complex128)
    n_ch, _, n_f = cov.shape

    mag = np.abs(cov)
    phase = np.angle(cov)

    mag_smooth = np.zeros_like(mag)
    for i in range(n_ch):
        for j in range(n_ch):
            log_mag = np.log(np.maximum(mag[i, j, :], psd_floor))
            log_mag_smooth = uniform_filter1d(
                log_mag,
                size=smooth_window_logfreq,
                mode="nearest",
            )
            mag_smooth[i, j, :] = np.exp(log_mag_smooth)

    cov_smooth = mag_smooth * np.exp(1j * phase)
    cov_stable = 0.5 * (cov_smooth + cov_smooth.conj().transpose(1, 0, 2))

    for k in range(n_f):
        d = np.real(np.diag(cov_stable[:, :, k]))
        d = np.maximum(d, psd_floor)
        np.fill_diagonal(cov_stable[:, :, k], d)

    if return_inverse:
        inv_cov_stable = np.empty_like(cov_stable)
        for k in range(n_f):
            inv_cov_stable[:, :, k] = np.linalg.inv(cov_stable[:, :, k])
        return cov_stable, inv_cov_stable

    return cov_stable


def inner_prod_tdi(a_fft, b_fft, cov_inv_matrices):
    a_fft_T = a_fft.T
    b_fft_T = b_fft.T
    inner_per_freq = np.einsum(
        "fj,fjk,fk->f",
        np.conj(a_fft_T),
        cov_inv_matrices,
        b_fft_T,
    )
    return 2 * np.real(np.sum(inner_per_freq))


def SNR(signal, invC):
    return np.sqrt(inner_prod_tdi(signal, signal, invC))


def match(s1, s2, invC):
    ip12 = inner_prod_tdi(s1, s2, invC)
    ip11 = inner_prod_tdi(s1, s1, invC)
    ip22 = inner_prod_tdi(s2, s2, invC)
    return ip12 / np.sqrt(ip11 * ip22)


def mismatch(s1, s2, invC):
    return 1 - match(s1, s2, invC)


def build_covariance_from_noise(noise_freqs, xyz_noise_estimate, target_freqs):
    n_f = len(target_freqs)
    cov = np.zeros((3, 3, n_f), dtype=complex)

    for i in range(3):
        for j in range(3):
            re = CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].real)(target_freqs)
            im = CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].imag)(target_freqs)
            cov[i, j, :] = re + 1j * im

    cov = 0.5 * (cov + cov.conj().transpose(1, 0, 2))
    for k in range(n_f):
        d = np.real(np.diag(cov[:, :, k]))
        np.fill_diagonal(cov[:, :, k], np.maximum(d, 1e-45))

    return cov


def load_noise_estimate():
    central_freq = 281600000000000.0
    with h5py.File(noise_file, "r") as f:
        xyz_noise_estimate = np.mean(f["noise_estimates/XYZ"][:], axis=0) / (central_freq ** 2)
        fmin_noise_psd = f["noise_estimates/log_frequency_sampling"].attrs["fmin"]
        fmax_noise_psd = f["noise_estimates/log_frequency_sampling"].attrs["fmax"]
        size_noise_psd = f["noise_estimates/log_frequency_sampling"].attrs["size"]
        noise_freqs = np.logspace(np.log10(fmin_noise_psd), np.log10(fmax_noise_psd), size_noise_psd)
    return noise_freqs, xyz_noise_estimate


def run_one_source(source_index, noise_freqs, xyz_noise_estimate):
    t_start_wall = time.time()
    result = {
        "source_index": source_index,
        "status": "ok",
        "snr_data": np.nan,
        "snr_template": np.nan,
        "snr_residual": np.nan,
        "mismatch_data_template": np.nan,
        "pdf_path": "",
        "error": "",
        "runtime_sec": np.nan,
    }

    pattern = os.path.join(SCRATCH, f"EMRI_731d_2.5s_L1_source{source_index}_*.h5")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No L1 file found for source {source_index} with pattern: {pattern}")
    L1_file_path = sorted(files)[-1]

    pdf_path = os.path.join(OUTPUT_DIR, f"EMRI_source_{source_index:02d}_report.pdf")
    result["pdf_path"] = pdf_path

    with MojitoL1File(L1_file_path) as f:
        tdi_time_sampling = f.tdis.time_sampling
        dt = tdi_time_sampling.dt
        central_freq = f.laser_frequency
        x2 = f.tdis.x2[:] / central_freq
        y2 = f.tdis.y2[:] / central_freq
        z2 = f.tdis.z2[:] / central_freq

    timing = get_mojito_timing(
        oemorbits_name,
        dt,
        t_dltt_orbits,
        dt_orbits,
        t_strain_offset,
        n_trim_lolipops,
        n_orbit_buffer,
        window_jaxgbresponse,
    )

    t_init = timing["t_init"]
    t0_orbits = timing["t0_orbits"]
    Tobs = 2.0
    n_samples = 1000
    offset = 550.0
    T_response = Tobs + (2 * offset + 2 * n_samples * dt) / ASTRONOMICAL_YEAR

    orbit_file = os.path.join(OUTPUT_DIR, "esa-trailing-orbits-mojito_validation.h5")
    if not os.path.exists(orbit_file):
        create_orbits(oemorbits_name, timing, orbit_file)

    esa = Orbits(
        filename=orbit_file,
        use_gpu=USE_GPU,
        force_backend=FORCE_BACKEND,
        linear_interp_setup=False,
        t0=t0_orbits,
    )

    emri_waveform = EMRIWaveBase(
        force_backend=FORCE_BACKEND,
        use_gpu=USE_GPU,
        inspiral_kwargs=inspiral_kwargs,
        sum_kwargs=sum_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        mode_selection_threshold=mode_selection_threshold,
        t0_orbits=t0_orbits,
        t_init=t_init,
    )

    params = get_source_params(
        "emri",
        source_id=source_index,
        username=MOJITO_USERNAME,
        token=MOJITO_TOKEN,
    )

    lam_ecl, beta_ecl = icrs_to_ecliptic(params["RightAscension"], params["Declination"])
    qS_ecl = np.pi / 2 - beta_ecl
    phiS_ecl = lam_ecl

    params_mojito = [
        params["PrimaryMassSSBFrame"],
        params["SecondaryMassSSBFrame"],
        params["PrimarySpinParameter"],
        params["SemiLatusRectum"],
        params["Eccentricity"],
        np.cos(params["InclinationAngle"]),
        params["LuminosityDistance"] * 1e-3,
        qS_ecl,
        phiS_ecl,
        params["PolarAnglePrimarySpin"],
        params["AzimuthalAnglePrimarySpin"],
        params["AzimuthalPhase"],
        params["PolarPhase"],
        params["RadialPhase"],
    ]

    emri_TDI_list = ResponseWrapper(
        emri_waveform,
        T_response,
        dt,
        index_lambda,
        index_beta,
        t0=t_init,
        t_buffer=10000.0,
        flip_hx=True,
        force_backend=FORCE_BACKEND,
        remove_sky_coords=False,
        is_ecliptic_latitude=False,
        remove_garbage=False,
        orbits=esa,
        **tdi_kwargs_esa_template,
    )

    chans = cp.asarray(emri_TDI_list(*params_mojito))
    tdi_channels_here = np.array([tdi_channel.get() for tdi_channel in chans])

    time_sim_L1 = np.arange(t_init + 850.5, x2.shape[0] * dt + t_init + 850.5, dt)[:-1]
    time_flr_L1 = np.arange(t_init, chans.shape[1] * dt + t_init, dt)[:-1]

    min_n = min(len(time_sim_L1), len(time_flr_L1), len(x2), tdi_channels_here.shape[1])
    time_sim_L1 = time_sim_L1[:min_n]
    time_flr_L1 = time_flr_L1[:min_n]
    x2 = x2[:min_n]
    y2 = y2[:min_n]
    z2 = z2[:min_n]
    tdi_channels_here = tdi_channels_here[:, :min_n]

    with PdfPages(pdf_path) as pdf:
        # Plot 1: Full channel overlays
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        sim_data = [x2, y2, z2]
        labels = ["X", "Y", "Z"]
        for i, lab in enumerate(labels):
            ax[i].plot(time_sim_L1, sim_data[i], label="SIM", alpha=0.85)
            ax[i].plot(time_flr_L1, tdi_channels_here[i], label="FLR", alpha=0.8)
            ax[i].set_title(f"TDI {lab} Channel")
            ax[i].set_xlabel("Time [s]")
            ax[i].set_ylabel("Amplitude")
            ax[i].legend(loc="upper left")
        fig.suptitle(f"Source {source_index} - Full Time-Domain Comparison")
        save_fig_to_pdf(pdf, fig)

        # Plot 2: Start zoom
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        for i, lab in enumerate(labels):
            ax[i].plot(time_sim_L1, sim_data[i], label="SIM")
            ax[i].plot(time_flr_L1, tdi_channels_here[i], label="FLR", alpha=0.85)
            ax[i].set_xlim(time_sim_L1[0] + 1000 * dt, time_sim_L1[0] + 8000 * dt)
            ax[i].set_title(f"{lab} Start Zoom")
            ax[i].set_xlabel("Time [s]")
            ax[i].legend(loc="upper left")
        fig.suptitle(f"Source {source_index} - Start Zoom")
        save_fig_to_pdf(pdf, fig)

        # Plot 3: Plunge zoom
        T_plunge = params["TimeCoalescenceSSBFrame"] + t_init - 50.5
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        for i, lab in enumerate(labels):
            ax[i].plot(time_sim_L1, sim_data[i], label="SIM")
            ax[i].plot(time_flr_L1, tdi_channels_here[i], label="FLR", alpha=0.85)
            ax[i].axvline(T_plunge, linestyle="--", color="black", label="Plunge")
            ax[i].set_xlim(0.99999 * T_plunge, 1.00001 * T_plunge)
            ax[i].set_title(f"{lab} Plunge Zoom")
            ax[i].set_xlabel("Time [s]")
            ax[i].legend(loc="upper left")
        fig.suptitle(f"Source {source_index} - Plunge Zoom")
        save_fig_to_pdf(pdf, fig)

        # Interpolate FLR channels on simulation grid
        xyz_splined = np.array(
            [CubicSpline(time_flr_L1, tdi_channels_here[i])(time_sim_L1) for i in range(3)]
        )

        # FFT diagnostics
        window = tukey(len(time_sim_L1), alpha=0.01)
        data = cp.asarray(np.array([x2, y2, z2]))
        model = cp.asarray(xyz_splined)

        xyz_residual = data - model
        xyz_data_windowed = data * cp.asarray(window)
        xyz_splined_windowed = model * cp.asarray(window)
        xyz_residual_windowed = xyz_residual * cp.asarray(window)

        xyz_data_fft = cp.fft.rfft(xyz_data_windowed, axis=1)
        xyz_splined_fft = cp.fft.rfft(xyz_splined_windowed, axis=1)
        xyz_residual_fft = cp.fft.rfft(xyz_residual_windowed, axis=1)

        N_t = len(x2)
        freqs = cp.fft.rfftfreq(N_t, d=dt)
        f_min = 1e-5
        f_max = 1.0 / (2.0 * dt)
        mask = (freqs >= f_min) & (freqs <= f_max)

        freqs_inband = freqs[mask]
        freqs_inband_np = np.asarray(freqs_inband.get())

        xyz_data_fft_inband = xyz_data_fft[:, mask]
        xyz_splined_fft_inband = xyz_splined_fft[:, mask]
        xyz_residual_fft_inband = xyz_residual_fft[:, mask]

        # Plot 4: PSD-like diagnostics
        channels = ["TDI X", "TDI Y", "TDI Z"]
        decim = 50
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for i, ch_name in enumerate(channels):
            axs[i].loglog(
                freqs_inband[::decim].get(),
                np.abs(xyz_data_fft_inband[i, ::decim].get()) ** 2,
                label="Data",
            )
            axs[i].loglog(
                freqs_inband[::decim].get(),
                np.abs(xyz_splined_fft_inband[i, ::decim].get()) ** 2,
                label="WF + FLR",
                ls="--",
                c="k",
                alpha=0.85,
            )
            axs[i].loglog(
                freqs_inband[::decim].get(),
                np.abs(xyz_residual_fft_inband[i, ::decim].get()) ** 2,
                label="Residual",
                c="gray",
                alpha=0.8,
            )
            axs[i].loglog(
                noise_freqs,
                np.abs(xyz_noise_estimate[:, i, i]),
                ls="dashed",
                c="red",
                label="Noise estimate",
            )
            axs[i].set_title(ch_name)
            axs[i].set_ylabel("ASD/PSD proxy")
            axs[i].legend(loc="lower left")
        axs[-1].set_xlabel("Frequency [Hz]")
        fig.suptitle(f"Source {source_index} - Frequency Diagnostics")
        save_fig_to_pdf(pdf, fig)

        # Build covariance and inverse
        covariance_matrices = build_covariance_from_noise(
            noise_freqs,
            xyz_noise_estimate,
            freqs_inband_np,
        )
        cov_stable, invC_stable = stabilize_covariance_for_inversion(
            covariance_matrices,
            psd_floor=1e-45,
            smooth_window_logfreq=15,
            return_inverse=True,
        )

        # Plot 5: Covariance diagnostics
        fig, axs = plt.subplots(3, 3, figsize=(11, 10), sharex=True)
        for i in range(3):
            for j in range(3):
                axs[i, j].loglog(freqs_inband_np[::5], np.abs(cov_stable[i, j, ::5]), label="abs")
                axs[i, j].loglog(freqs_inband_np[::5], np.abs(np.real(cov_stable[i, j, ::5])), label="real")
                axs[i, j].loglog(freqs_inband_np[::5], np.abs(np.imag(cov_stable[i, j, ::5])), label="imag")
                axs[i, j].grid(True, which="both", alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(loc="lower left")
        fig.suptitle(f"Source {source_index} - Covariance Matrix Diagnostics")
        save_fig_to_pdf(pdf, fig)

        # Scalar diagnostics
        pre_fact = 2.0 * dt / N_t
        invC_for_ip = pre_fact * invC_stable
        invC = np.transpose(invC_for_ip, (2, 0, 1))

        s_data = np.asarray(xyz_data_fft_inband.get())
        s_model = np.asarray(xyz_splined_fft_inband.get())
        s_res = np.asarray(xyz_residual_fft_inband.get())

        snr_data = SNR(s_data, invC)
        snr_template = SNR(s_model, invC)
        snr_residual = SNR(s_res, invC)
        mm = mismatch(s_data, s_model, invC)

        result["snr_data"] = float(snr_data)
        result["snr_template"] = float(snr_template)
        result["snr_residual"] = float(snr_residual)
        result["mismatch_data_template"] = float(mm)

        # Plot 6: Text summary page
        fig = plt.figure(figsize=(11.69, 8.27))
        txt = (
            f"Source {source_index} Summary\n\n"
            f"L1 file: {L1_file_path}\n"
            f"dt: {dt}\n"
            f"N samples (used): {N_t}\n\n"
            f"SNR data:      {snr_data:.8e}\n"
            f"SNR template:  {snr_template:.8e}\n"
            f"SNR residual:  {snr_residual:.8e}\n"
            f"Mismatch:      {mm:.8e}\n\n"
            f"t_init:        {t_init:.6f}\n"
            f"t0_orbits:     {t0_orbits:.6f}\n"
            f"T_plunge est:  {T_plunge:.6f}\n"
        )
        fig.text(0.05, 0.95, txt, va="top", family="monospace", fontsize=12)
        fig.suptitle(f"Source {source_index} - Numerical Diagnostics")
        save_fig_to_pdf(pdf, fig)

    result["runtime_sec"] = time.time() - t_start_wall
    return result


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    noise_freqs, xyz_noise_estimate = load_noise_estimate()

    results = []
    for source_index in SOURCE_INDICES:
        print(f"\nRunning source {source_index} ...")
        try:
            r = run_one_source(source_index, noise_freqs, xyz_noise_estimate)
        except Exception as exc:
            print(f"Error processing source {source_index}: {exc}")
            r = {
                "source_index": source_index,
                "status": "failed",
                "snr_data": np.nan,
                "snr_template": np.nan,
                "snr_residual": np.nan,
                "mismatch_data_template": np.nan,
                "pdf_path": "",
                "error": str(exc),
                "runtime_sec": np.nan,
            }
        results.append(r)

    df = pd.DataFrame(results).sort_values("source_index").reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "emri_validation_summary.csv")
    json_path = os.path.join(OUTPUT_DIR, "emri_validation_summary.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    # Structured terminal output
    print("\n" + "=" * 90)
    print("EMRI VALIDATION SUMMARY")
    print("=" * 90)
    display_cols = [
        "source_index",
        "status",
        "snr_data",
        "snr_template",
        "snr_residual",
        "mismatch_data_template",
        "runtime_sec",
        "pdf_path",
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    failed = df[df["status"] != "ok"]
    if len(failed) > 0:
        print("\nFAILED SOURCES")
        print(failed[["source_index", "error"]].to_string(index=False))
    print("\nSaved:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- Per-source PDFs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
