'''
Code to validate the Mojito dataset with PE using fast template
This is also a test for the FEW review.
'''

# imports
# imports
import os
os.environ["GPUBACKENDTOOLS_FORCE_BACKEND"] = "cuda12x"

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from few.waveform import GenerateEMRIWaveform
from fastlisaresponse.utils.parallelbase import ParallelModuleBase
from lisatools.detector import Orbits
from lisaconstants import ASTRONOMICAL_YEAR

import numpy as np
import cupy as cp
import os
import h5py
from scipy.signal.windows import tukey
from cupyx.scipy.interpolate import CubicSpline

from h5py import File
from lisaconstants import ASTRONOMICAL_YEAR
from lisaorbits import OEMOrbits


YRSID_SI = ASTRONOMICAL_YEAR

import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import logging
# Set logging level to INFO (more verbose and informative)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import h5py

from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()
my_password = os.getenv("LISA_CONSORTIUM_KEY")
my_username = os.getenv("LISA_CONSORTIUM_NAME")

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend
# need to install the Mojito package for reading in parameters and data
from mojito import MojitoL1File   
from mojito.download import get_source_params

# read in source index from parser
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, help = "WHich source to sample", default=0)
args = parser.parse_args()
source_index = args.source

import few
import lisatools
import fastlisaresponse
import gpubackendtools
logger.info(f''' Package versions:
few:              {few.__version__}
lisatools:        {lisatools.__version__}
fastlisaresponse: {fastlisaresponse.__version__}
gpubackendtools:  {gpubackendtools.__version__}
''')

# fetch correct dataset
import glob
import os

# change this to dataset location
# scratch = '/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/mojito_light_v1_0_0/data/EMRI/L1'
scratch = '/scratch/project_2004833/common_data/mojito/brickmarket/mojito_v1_0_0/data/EMRI/L1_0p4Hz'

# Use * as a wildcard for the parts that change
pattern = os.path.join(scratch, f'EMRI_731d_2.5s_L1_source{source_index}_*.h5')
files = glob.glob(pattern)
output = f'{os.getcwd()}/output'
if files:
    # If multiple versions, sort on stimestamps
    L1_file_path = sorted(files)[-1] 
    print(f"Selected: {L1_file_path}")
else:
    print("No file found for that source index.")
 
with MojitoL1File(L1_file_path) as f:
    tdi_time_sampling = f.tdis.time_sampling
    tdi_dt = tdi_time_sampling.dt
    
    # TDI observables
    CENTRAL_FREQ = f.laser_frequency
    
    x2 = f.tdis.x2[:] / CENTRAL_FREQ  # TDI X2 observable in Hz
    y2 = f.tdis.y2[:] / CENTRAL_FREQ # TDI Y2 observable in Hz
    z2 = f.tdis.z2[:] / CENTRAL_FREQ # TDI Z2 observable in Hz
   
# timings
dt = delta_t = tdi_dt
len_waveform = int(x2.shape[0] + 2* (550/dt + 1000))
Time = len_waveform / ASTRONOMICAL_YEAR *dt
oemorbits =  "esa-trailing"
t_dltt_orbits = 10.

t0_l1 = tdi_time_sampling.t0
t0_l0 = t0_l1 - 1000*dt
t_init = t0_l0 - 550

# waveform
# Set up correct arguments
sum_kwargs = {
    "pad_output": True,
}

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # sparsely sampled trajectory
    "max_init_len": int(1e8),  
}

amplitude_kwargs = {
    # "max_init_len": int(1e8),  # all of the trajectories will be well under len = 1000
    # "use_gpu": True,
    # "file_dir":"/data/leuven/367/vsc36785/LISA/FastEMRIWaveforms/data"
}

waveform_kwargs = {
    'mode_selection_threshold': 0.0
}

class EMRIWave_base(ParallelModuleBase):
    def __init__(self, force_backend=None, 
                use_gpu=True, 
                 inspiral_kwargs=inspiral_kwargs,
                 sum_kwargs=sum_kwargs,
                 amplitude_kwargs=amplitude_kwargs,
                 mode_selection_threshold=1e-5,
                 t_init=33568152.5,
                 t0_orbits=33568152.5,
                 dt=5, 
                 n_samples=1000,
                 offset=550, # seconds
                 time=2.0
                ):
                 
        super().__init__(force_backend=force_backend)
        
        self.use_gpu = use_gpu
        self.mode_threshold = mode_selection_threshold
        
        # Initialize waveform generator

        self.waveform_gen = GenerateEMRIWaveform(
                "FastKerrEccentricEquatorialFlux",
                return_list=False,    # returns hp - i*hx as a complex cupy array
                inspiral_kwargs=inspiral_kwargs,
                sum_kwargs=sum_kwargs,
                amplitude_kwargs=amplitude_kwargs,
                frame="detector"
            )
        self.t_init = t_init
        self.t0_orbits = t0_orbits
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, *params, T=2, dt=5):
        '''
        Call FEW waveform model and return the strain as h_+ - ih_x
        '''
        # define correct time grid for waveform generation.
        waveform_kwargs['T'] = T
        waveform_kwargs['dt'] = dt
        waveform_kwargs['mode_selection_threshold'] = self.mode_threshold
        strain = self.waveform_gen(*params, **waveform_kwargs)

        return strain
    
# get t0_orbits as done in Mojito simulation
orbits = OEMOrbits.from_included(oemorbits)
t0_orbits = float(orbits.t_start) + t_dltt_orbits

n_samples = 1000 # necessary to do the TDI delays
Tobs = 2 # years
offset = 550  # seconds

T = Tobs + (2* offset + 2*n_samples*dt)/ASTRONOMICAL_YEAR

use_gpu=True
t_smooth=0
waveform_model='Kerr'
mode_selection_threshold = 0.0

f_s = 1/dt   
home_folder = os.getcwd()


emri_waveform = EMRIWave_base(use_gpu=use_gpu, 
                         mode_selection_threshold=mode_selection_threshold,
                         t0_orbits=t0_orbits,
                         t_init=t_init,
                         dt=dt, 
                         n_samples=n_samples,
                         offset=offset, # seconds
                        )

# Get source parameters for mbhb brick, source ID 12
params = get_source_params("emri", source_id=source_index, username=my_password, token=my_username)
for key in params.keys():
    logger.info(f'{key} = %d', params[key])
    
def icrs_to_ecliptic(ra, dec):
    """Convert ICRS coordinates (ra, dec) to ecliptic coordinates (lambda, beta)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    icrs_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
    ecliptic_coord = icrs_coord.barycentrictrueecliptic

    lambda_ecl = ecliptic_coord.lon.rad
    beta_ecl = ecliptic_coord.lat.rad

    return lambda_ecl, beta_ecl

ra = params['RightAscension']
dec = params['Declination']

lam_ecl, beta_ecl = icrs_to_ecliptic(ra, dec)
qS_ecl = np.pi / 2 - beta_ecl
phiS_ecl = lam_ecl

# create array with True parameters
params_mojito = [
    params['PrimaryMassSSBFrame'],
    params['SecondaryMassSSBFrame'],
    params['PrimarySpinParameter'], #* np.sign(np.cos(params['InclinationAngle'])),
    params['SemiLatusRectum'],
    params['Eccentricity'],
    np.cos(params['InclinationAngle']),
    params['LuminosityDistance']*1e-3,
    qS_ecl,  # np.pi/2 - beta
    phiS_ecl,  # lambda
    params['PolarAnglePrimarySpin'],
    params['AzimuthalAnglePrimarySpin'],
    params['AzimuthalPhase'],
    params['PolarPhase'],
    params['RadialPhase'],
]

# response
home_folder = os.getcwd()
orbit_file= f'{home_folder}/esa-trailing-orbits-mojito_validation_test_2.h5'
force_backend = 'cuda12x'
esa = Orbits(filename=orbit_file, 
            use_gpu=use_gpu, 
            force_backend=force_backend, 
            linear_interp_setup=False,
            t0=t0_orbits)

force_backend = "cuda12x" if use_gpu else None

index_beta = 7
index_lambda = 8

tdi_kwargs = {
    'tdi': '2nd generation',
    'tdi_chan': 'XYZ',
    'order': 39,
}

tdi_kwargs_esa = dict(
            orbits=esa,
            order=40,
            tdi=TDIConfig('2nd generation'),
            tdi_chan="XYZ",
        )

emri_TDI_list = ResponseWrapper(
    emri_waveform,
    T,
    dt,
    index_lambda,
    index_beta,
    t0=t_init,
    t_buffer = 10000.0,
    flip_hx=True,  # set to True if waveform is h+ - ihx
    force_backend=force_backend,
    remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=False,  # False if using polar angle (theta)
    remove_garbage=False,  # removes the beginning of the signal that has bad information
    # orbits=esa,
    **tdi_kwargs_esa,
)

def emri_TDI(*params):
    return cp.asarray(emri_TDI_list(*params))

# test
chans = emri_TDI(*params_mojito)

# Process Mojito data onto correct time array
time_sim_L1 = np.arange(t_init + 850.5, x2.shape[0]*delta_t + t_init + 850.5, dt)[:-1]  
time_flr_L1 = np.arange(t_init, chans.shape[1]*delta_t + t_init, dt)[:-1] 

data = cp.asarray([x2, y2, z2])

# from scipy.interpolate import CubicSpline
window = cp.asarray(tukey(len(time_sim_L1), alpha=0.01))

# This data array is splined onto the same time array that is returned by fastlisaresponse. 
# Otherwise no perfect subtraction possible
data_splined = CubicSpline(time_sim_L1, data, axis=1)(time_flr_L1) 

# Compute residual
xyz_residual = data_splined - chans
xyz_residual_windowed = xyz_residual * window

N_t = len(data[0])
freqs = cp.fft.rfftfreq(N_t, d=dt)

f_max = 1/(2*dt) # Nyquist frequency
f_min = 1e-5 # minimum frequency to consider
mask = (freqs >= f_min) & (freqs <= f_max)
freqs_inband = freqs[mask]

xyz_residual_windowed_fft = cp.fft.rfft(xyz_residual_windowed, axis=1)[:, mask]
xyz_data_fft = cp.fft.rfft(data_splined * window, axis=1)[:, mask]
xyz_template_fft = cp.fft.rfft(chans * window, axis=1)[:, mask]

# Load and process noise model
noise_file = f"/scratch/project_2004833/common_data/mojito/brickmarket/mojito_v1_0_0/data/INSTRUMENT/L1_0p4Hz/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"

with h5py.File(noise_file, "r") as f:
    xyz_noise_estimate = np.mean(f['noise_estimates/XYZ'][:], axis=0) / CENTRAL_FREQ**2
    fmin_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['fmin']
    fmax_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['fmax']
    size_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['size']

    noise_freqs = np.logspace(np.log10(fmin_noise_psd), np.log10(fmax_noise_psd), size_noise_psd)

# Interpolate noise curves
freqs_inband_np = np.asarray(freqs_inband.get())

splined_noise_psd = cp.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, i])(freqs_inband_np) for i in range(3)
])

splined_noise_csd_real = cp.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].real)(freqs_inband_np) for i in range(3) for j in range(i, 3)
])

splined_noise_psd_imag = cp.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].imag)(freqs_inband_np) for i in range(3) for j in range(i, 3)   
])


# now re-assemble the covariance matrix
covariance_matrices = np.zeros((3, 3, len(freqs_inband)), dtype=complex)
for i in range(3):
    covariance_matrices[i, i, :] = splined_noise_psd[i]
    for j in range(i+1, 3):
        covariance_matrices[i, j, :] = splined_noise_csd_real[i*3 + j - (i+1)*i//2] + 1j * splined_noise_psd_imag[i*3 + j - (i+1)*i//2]
        covariance_matrices[j, i, :] = np.conj(covariance_matrices[i, j, :])

invC = np.linalg.inv(covariance_matrices.transpose(2, 0, 1))
pre_fact = 2 * dt / N_t
invC *= pre_fact


def check_memory():
    free, total = cp.cuda.Device(0).mem_info
    print(f'Free memory  : {free/1e9:.2f} Gb\nUsed memory  : {(total-free)/1e9:.2f} Gb\nTotal memory : {total/1e9:.2f} Gb\n')
check_memory()

def inner_prod_tdi(a_fft, b_fft, cov_inv_matrices):
    """
    Compute noise-weighted inner product for TDI multi-channel data with correlated noise.
    Time-Delay Interferometry (TDI) produces three channels (X, Y, Z) with correlated
    noise. This function computes the proper inner product accounting for these
    correlations using the inverse covariance matrix.
    Mathematical form:
        ⟨a|b⟩ = 2 Re[Σₖ aₖ† Σₖ⁻¹ bₖ]
    where at each frequency bin k:
        - aₖ, bₖ are 3D complex vectors [X_k, Y_k, Z_k]
        - Σₖ⁻¹ is the 3×3 inverse covariance matrix
    Parameters
    ----------
    a_fft : array-like, complex, shape (3, n_freq)
        First frequency-domain TDI waveform [X, Y, Z] × frequencies
    b_fft : array-like, complex, shape (3, n_freq)
        Second frequency-domain TDI waveform [X, Y, Z] × frequencies
    cov_inv_matrices : array-like, float, shape (n_freq, 3, 3)
        Inverse covariance matrices for each frequency bin
        Pre-multiplied by prefactor (typically 2*dt/N)
    Returns
    -------
    float
        Noise-weighted inner product accounting for correlations between channels
    Notes
    -----
    - Uses Einstein summation (einsum) for efficient vectorized computation
    - At each frequency: result[k] = conj(a[k]).T @ Σ⁻¹[k] @ b[k]
    - The covariance matrix is NOT diagonal due to TDI channel correlations
    - Essential for computing SNRs and Fisher matrices in TDI analysis
    Implementation Details
    ----------------------
    The computation uses Einstein notation 'fj,fjk,fk->f' which means:
        f : frequency index
        j : first TDI channel index (for a)
        k : second TDI channel index (for b)
    This computes: Σⱼₖ conj(a[f,j]) * Σ⁻¹[f,j,k] * b[f,k] for each frequency
    See Also
    --------
    inner_prod : Simpler inner product for single-channel data
    """
    # Reshape from (3, n_freq) to (n_freq, 3) to align with covariance matrix indexing
    # This makes frequency the leading dimension for efficient vectorized operations
    a_fft_T = a_fft.T  # Shape: (n_freq, 3) - [X, Y, Z] for each frequency
    b_fft_T = b_fft.T  # Shape: (n_freq, 3) - [X, Y, Z] for each frequency
    # Compute matrix product for each frequency: aₖ† Σₖ⁻¹ bₖ
    # Einstein summation notation 'fj,fjk,fk->f' performs:
    #   f = frequency index (output dimension)
    #   j = TDI channel for left vector (a)
    #   k = TDI channel for right vector (b)
    #
    # For each frequency f, computes:
    #   Σⱼ Σₖ conj(a[f,j]) * Σ⁻¹[f,j,k] * b[f,k]
    # which is equivalent to the matrix product: a†[f] @ Σ⁻¹[f] @ b[f]
    inner_per_freq = cp.einsum('fj,fjk,fk->f',
                               np.conj(a_fft_T),
                               cov_inv_matrices,
                               b_fft_T)
    # Sum over all frequencies and extract real part
    # Factor of 2 accounts for positive and negative frequencies (one-sided spectrum)
    inner_product = 2 * cp.real(cp.sum(inner_per_freq))
    return inner_product

# likelihood
def llike(params):
    """
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    """
    # define params with constant inclination and Polar phase
    # Intrinsic Parameters
    M_val = float(params[0])
    mu_val =  float(params[1])
    a_val =  float(params[2])            
    p0_val = float(params[3])
    e0_val = float(params[4])
    x_I0_val = params_mojito[6]
    
    # Luminosity distance 
    D_val = float(params[5])

    # Angular Parameters
    qS_val = float(params[6])
    phiS_val = float(params[7])
    qK_val = float(params[8])
    phiK_val = float(params[9])

    # Angular parameters
    Phi_phi0_val = float(params[10])
    Phi_theta0_val = params_mojito[12]
    Phi_r0_val = float(params[11])

    waveform_prop = emri_TDI(*[M_val, mu_val, a_val, p0_val, e0_val, 
                                  x_I0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0_val, Phi_theta0_val, Phi_r0_val])  # EMRI waveform across X, Y, Z.

    # Compute in frequency domain
    EMRI_XYZ_fft_prop = cp.fft.rfft(waveform_prop * window, axis=1)[:, mask] 
    
    # Compute (d - h| d- h)
    diff_f_XYZ = xyz_residual_windowed_fft - EMRI_XYZ_fft_prop
    
    inn_prod = inner_prod_tdi(
        diff_f_XYZ,
        diff_f_XYZ, 
        invC)
                         
    
    # Return log-likelihood value as numpy val. 
    llike_val_np = cp.asnumpy(-0.5 * inn_prod) 
    return (llike_val_np)

# Perform sanity checks and make plots as in notebook
def SNR(signal, invC):
    return np.sqrt(inner_prod_tdi(signal, signal, invC))

def match(s1, s2, invC):
    ip12 = inner_prod_tdi(s1, s2, invC)
    ip11 = inner_prod_tdi(s1, s1, invC)
    ip22 = inner_prod_tdi(s2, s2, invC)
    return ip12 / np.sqrt(ip11 * ip22)

def mismatch(s1, s2, invC):
    return 1 - match(s1, s2, invC)

SNR_data = SNR(xyz_data_fft, invC)
SNR_wf = SNR(xyz_template_fft, invC)
SNR_residual = SNR(xyz_residual_windowed_fft, invC)
mismatch = mismatch(xyz_template_fft, xyz_data_fft, invC)
logging.info(f"SNR of data:     {SNR_data:.2f}")
logging.info(f"SNR of template: {SNR_wf:.2f}")
logging.info(f"SNR of residual: {SNR_residual:.2f}")
logging.info(f"Mismatch between template and data: {mismatch:.2e}")

logging.info("Starting PE run: setting up parameters")
iterations = 20000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains
d = 0.1
nwalkers = 128
ntemps = 1        
Reset_Backend = True
tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

logger.info(f"PE run settings: ")
logger.info(f"iterations={iterations}")
logger.info(f"burnin={burnin}")
logger.info(f"nwalkers={nwalkers}")
logger.info(f"ntemps={ntemps}")
logger.info(f"Reset_backend={Reset_Backend}")
logger.info(f"tempering_kwargs={tempering_kwargs}")
logger.info(f"d = {d}")
logger.infor("Setting up priors and initial values...")
# set priors
priors_in = {
    # Intrinsic parameters
    0: uniform_dist(params_mojito[0]*0.9999, params_mojito[0]*1.0001), # Primary Mass M
    1: uniform_dist(params_mojito[1]*0.9999, params_mojito[1]*1.0001), # Secondary Mass mu
    2: uniform_dist(params_mojito[2]*0.9999, params_mojito[2]*1.0001), # Spin parameter a
    3: uniform_dist(params_mojito[3]*0.9999, params_mojito[4]*1.0001), # semi-latus rectum p0
    4: uniform_dist(params_mojito[4]*0.9999, params_mojito[5]*1.0001), # eccentricity e0
    5: uniform_dist(params_mojito[6]*0.9, params_mojito[6]*1.1), # distance D
    # Extrinsic parameters -- Angular parameters
    6: uniform_dist(0, np.pi), # Polar angle (sky position)
    7: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    8: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    9: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    10: uniform_dist(0, 2*np.pi), # Phi_phi0
    11: uniform_dist(0, 2*np.pi) # Phi_r0
}  


# set initialization for PE run
# This must be extremely close to the true values
# We start the sampler exceptionally close to the true parameters and let it run. This is reasonable 
# if and only if we are quantifying how well we can measure parameters. We are not performing a search. 
logger.info("Setting up initial values for the sampler...")
# Intrinsic Parameters

start_M = params_mojito[0]*(1. + d * 1e-7 * np.random.randn(nwalkers,1))   
start_mu = params_mojito[1]*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_a = params_mojito[2]*(1. + d * 1e-7 * np.random.randn(nwalkers,1))

start_p0 = params_mojito[3]*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))
start_e0 = params_mojito[4]*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

# Luminosity distance
start_D = params_mojito[6]*(1 + d * 1e-6 * np.random.randn(nwalkers,1))

# Angular parameters
start_qS = params_mojito[7]*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = params_mojito[8]*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = params_mojito[9]*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = params_mojito[10]*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

# Initial phases 
start_Phi_Phi0 = params_mojito[11]*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_r0 = params_mojito[13]*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_mu, start_a, start_p0, start_e0, start_D, 
start_qS, start_phiS, start_qK, start_phiK,start_Phi_Phi0, start_Phi_r0))

if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other. 
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

priors = ProbDistContainer(priors_in, use_cupy = True)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================
moves_stretch = StretchMove(a=2.0, use_gpu=True)

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0])) 
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0])) 

logger.info("Setting up backend  and sampler...")
data_dir = f'{os.getcwd()}/data' 
fp = f"{data_dir}/PE_run_source_{source_index}.h5"
backend = HDFBackend(fp)
logger.info(f"Backend set up at {fp}")

ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch,
                            vectorize=True
                            )


if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch,
                            vectorize=True
                            )
else:
    start = backend.get_last_sample() # Start from last sample
logger.info("Starting MCMC sampling...")
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler
