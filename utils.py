import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time
import re

label = "gaussian_with_dm"
outdir = "/fred/oz005/users/vdimarco/Portraiture/outdir"
plotdir = "/fred/oz005/users/vdimarco/Portraiture/plots"
# Create directories if they don't exist
os.makedirs(outdir, exist_ok=True)
os.makedirs(plotdir, exist_ok=True)

n = 10  # number of frequency bins
freqs = np.linspace(600, 1000, n).astype(int).tolist() # Radio frequencies in MHz
#freqs = [600, 1000, 1400, 1800, 2200, 2600, 3000, 3200] # Radio frequencies in MHz
print(freqs)
n_points = 100
n_pulses = 5
base_dm = 10.0  # typical for a Galactic pulsar between 
variation_std = 0.0001  # small DM variations

K_DM = 4.15e3  # MHz^2*pc^-1*cm^3 [1/s^2 * pc^-1*cm^3 ]

#define parameters for function FRED
Delta = 0.4 # time delay
A = 2 # this scales the function
xi = 1 # asimmmetry parameter
tau = 0.20 # duration scaling parameter
gamma = 2.5 # exponent for flatter/sharper peak
nu = 1 # exponent for flatter/sharper peak
sigma = 0.02

#######################
### FUNCTIONS
#######################

# Function to add white noise to a signal
def add_white_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

# Function to create a Lorentzian function
def create_lorentzian(n_points=100, mean=0.5, gamma=0.05, noise_level=0.1):
    x = np.linspace(0, 1, n_points)
    lorentzian = 1 / (1 + ((x - 0.5) / gamma) ** 2)
    # Lorentzian function: f(x) = 1 / (1 + ((x - mean) / gamma) ** 2)
    noisy_lorentzian = add_white_noise(lorentzian, noise_level)
    return x, noisy_lorentzian

# Function to create a Gaussian function
def create_gaussian(n_points=100, mean=0.5, sigma=0.05, noise_level=0.1): 
    x = np.linspace(0, 1, n_points)
    gaussian = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2) 
    noisy_gaussian = add_white_noise(gaussian, noise_level)
    return x, noisy_gaussian

# define FRED (fast rise exponential decay) function
def fred(A, xi, phi, Delta, tau):
    shifted_phi = phi - Delta # apply phase shift and Delta
    shifted_phi = np.where(np.abs(shifted_phi) < 1e-10, 1e-10, shifted_phi)  # avoid division by zero
    term_1 = shifted_phi / tau  # first term in the exponent
    term_2 = tau / shifted_phi  # second term in the exponent
    exponent = -xi**gamma * (term_1**gamma + term_2**nu)  # full exponent
    FRED = A * np.exp(exponent)  # FRED function result
    FRED_cleaned = np.nan_to_num(FRED, nan=0.0)  # remove NaNs
    return FRED_cleaned # add white noise

# Function to apply a DM-induced delay
def apply_dm_delay(signal, dm, freq, n_points):
    #[1/s^2 * pc^-1*cm^3 ] * pc/cm^3 = 1/s^2
    # K_DM Constant for DM delay calculation
    delay = int(np.round(K_DM * dm / freq**2 * n_points))  # Delay in samples
    delay = (delay%n_points)/n_points
    return translate_signal(signal, delay)

# Function to invert the DM-induced delay
def invert_dm_delay(signal, dm, freq, n_points):
    delay = int(np.round(K_DM * dm / freq**2 * n_points))  # Delay in samples
    delay = (delay%n_points)/n_points
    return translate_signal(signal, -delay)

# Function to translate a signal in time
def translate_signal(signal, shift):
    n = len(signal)
    shift = int(shift*n)
    
    shifted_signal = np.zeros_like(signal)
    if shift > 0:
        shifted_signal[shift:] = signal[:n - shift]
        shifted_signal[:shift] = signal[n - shift:]
    elif shift < 0:
        shift = -shift
        shifted_signal[:n - shift] = signal[shift:]
        shifted_signal[n - shift:] = signal[:shift]
    else:
        shifted_signal = signal.copy()
    return shifted_signal

# Function to generate signals with shifts and DMs
def generate_signals(shifts, dms, freqs, n_points):
    n_signals = len(shifts)
    n_freqs = len(freqs)

    original = np.zeros((n_signals, n_freqs, n_points))
    shifted = np.zeros((n_signals, n_freqs, n_points))

    x = np.linspace(0, 1, n_points)

    for i in range(n_signals):
        for j in range(n_freqs):
            # Use Gaussian for even i, Lorentzian for odd i
            # for a bit of variety :-)
            # if i % 2 == 0:
            #     _, pulse = create_gaussian(n_points, mean=0.5, sigma=0.05)
            # else:
            #     _, pulse = create_lorentzian(n_points, mean=0.5, gamma=0.02)

            _, pulse = create_gaussian(n_points, mean=0.5, sigma=0.05)

            original[i, j] = pulse

            # Shift in time (horizontal shift)
            shifted_pulse = translate_signal(pulse, shifts[i])

            # Dispersion (vertical shift across frequency)
            dispersed_pulse = apply_dm_delay(shifted_pulse, dms[i], freqs[j], n_points)

            shifted[i, j] = dispersed_pulse

    return original, shifted

# Function to model the signal
def model_signal(*params, shifted_datasets):
    n_signals = shifted_datasets.shape[0]
    n_freqs = shifted_datasets.shape[1]
    n_points = shifted_datasets.shape[2]

    dms = params[:n_signals]
    shifts = params[n_signals:]

    modeled_signals = []
    total_aligned_signals = []
    for i in range(n_signals):  # model each signal i
        # Indices of other signals (to build template)
        conditioning_indices = list(range(i+1, n_signals))
        print("conditioning_indices: ", conditioning_indices)
        other_indices = [j for j in range(n_signals) if j != i]
        #print("index i: ", i)
        #print("other_indices: ", other_indices)

        # Build template across bands
        
        templates = []
        all_aligned_signals = []
        for f in range(n_freqs):
            # Undo DM + time shifts for all other signals
            aligned_signals = []
            for j in other_indices:
            #for j in conditioning_indices:
                undispersed = invert_dm_delay(
                    shifted_datasets[j, f], dms[j], freqs[f], n_points)
                aligned = translate_signal(undispersed, -shifts[j])
                aligned_signals.append(aligned)
            
            # Average template across other signals
            template = np.mean(aligned_signals, axis=0)
            templates.append(template)
            all_aligned_signals.append(template)
        print("templates: ", np.shape(templates))
        total_aligned_signals.append(templates)
        # Reapply DM and shift to simulate signal i
        band_signals = []
        for f, freq in enumerate(freqs):
            redispersed = apply_dm_delay(templates[f], dms[i], freq, n_points)
            shifted = translate_signal(redispersed, shifts[i])
            band_signals.append(shifted)

        modeled_signals.append(band_signals)
    print("total_aligned_signals: ", np.shape(total_aligned_signals))

    return np.array(total_aligned_signals), np.array(modeled_signals)  # shape: (n_signals, n_freqs, n_points)

# # Funtion to plot model and signal
# def plot_it(modeled_signals,datasets):
#     x = np.linspace(0, 1, datasets.shape[2])
#     n_pulses = datasets.shape[0]
#     n_freqs = datasets.shape[1]
#     offset_step = 1
#     y_ticks = [i * offset_step for i in range(n_freqs)]
#     y_labels = [f"{freq} MHz" for freq in freqs]
#     plt.figure(figsize=(12, 8))
#     fig, axes = plt.subplots(n_pulses, 1, figsize=(12, 4 * n_pulses), sharex=True)
#     for k in range(n_pulses):
#         ax = axes[k] if n_pulses > 1 else axes  # handle 1 subplot edge case
#         for band in range(n_freqs):
#             offset = band * offset_step
#             y_data_raw = datasets[k, band]  + offset
#             y_model = modeled_signals[k, band]  + offset
#             # Plot raw data in grey
#             ax.plot(x, y_data_raw, color='grey', label=f"Data {freqs[band]} MHz" if k == 0 else "")
#             # Plot model in dark blue
#             ax.plot(x, y_model, color='darkblue', linestyle='--', label=f"Model {freqs[band]} MHz" if k == 0 else "")
#             ax.set_ylabel(f"{freqs[band]} MHz")
#         ax.set_title(f"Portrait {k+1}")
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(y_labels)
#         ax.set_title(f"Portrait {k + 1}")
#         ax.set_ylabel("Frequency")
#     # Common x-axis label at bottom
#     axes[-1].set_xlabel("Phase")
#     # Shared legend (optional)
#     handles, labels = axes[0].get_legend_handles_labels()
#     custom_lines = [
#         Line2D([0], [0], color='grey', label='Data'),
#         Line2D([0], [0], color='darkblue', linestyle='--', label='Template')
#     ]
#     fig.legend(handles=custom_lines, loc='upper right')
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     fig.suptitle("Portraits", fontsize=16)
#     plt.savefig(f"{plotdir}/model_fit.png")

# Function to plot model and signal, saving one plot per pulse
def plot_it(modeled_signals, datasets):
    x = np.linspace(0, 1, datasets.shape[2])
    n_pulses = datasets.shape[0]
    n_freqs = datasets.shape[1]
    offset_step = 1
    y_ticks = [i * offset_step for i in range(n_freqs)]
    y_labels = [f"{freq} MHz" for freq in freqs]
    for k in range(n_pulses):
        fig, ax = plt.subplots(figsize=(12, 4))
        for band in range(n_freqs):
            offset = band * offset_step
            y_data_raw = datasets[k, band] + offset
            y_model = modeled_signals[k, band] + offset
            ax.plot(x, y_data_raw, color='grey', label=f"Data {freqs[band]} MHz" if band == 0 else "")
            ax.plot(x, y_model, color='darkblue', linestyle='--', label=f"Model {freqs[band]} MHz" if band == 0 else "")
        ax.set_title(f"Portrait {k + 1}")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Phase")
        # Add shared legend
        custom_lines = [
            Line2D([0], [0], color='grey', label='Data'),
            Line2D([0], [0], color='darkblue', linestyle='--', label='Template')
        ]
        ax.legend(handles=custom_lines, loc='upper right')
        fig.tight_layout()

        plt.savefig(f"{plotdir}/model_fit_pulse_{k + 1}.png")
        plt.close(fig)