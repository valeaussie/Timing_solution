import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time
import re
import shutil

plt.close('all')

label = "gaussian_with_dm"
outdir = "/fred/oz005/users/vdimarco/Portraiture/outdir"
plotdir = "/fred/oz005/users/vdimarco/Portraiture/plots"
# Create directories if they don't exist
os.makedirs(outdir, exist_ok=True)
os.makedirs(plotdir, exist_ok=True)

freqs = [600, 1400, 2100, 3200]  # Radio frequencies in MHz
n_points = 100
n_pulses = 4
base_dm = 10.0  # typical for a Galactic pulsar between 
variation_std = 0.0001  # small DM variations

K_DM = 4.15e3  # MHz^2*pc^-1*cm^3 [1/s^2 * pc^-1*cm^3 ]

def add_white_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def create_lorentzian(n_points=100, mean=0.5, gamma=0.05, noise_level=0.1):
    x = np.linspace(0, 1, n_points)
    lorentzian = 1 / (1 + ((x - 0.5) / gamma) ** 2)
    # Lorentzian function: f(x) = 1 / (1 + ((x - mean) / gamma) ** 2)
    noisy_lorentzian = add_white_noise(lorentzian, noise_level)
    return x, noisy_lorentzian

def create_gaussian(n_points=100, mean=0.5, sigma=0.05, noise_level=0.1): 
    x = np.linspace(0, 1, n_points)
    gaussian = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2) 
    noisy_gaussian = add_white_noise(gaussian, noise_level)
    return x, noisy_gaussian

# Function to apply a DM-induced delay
def apply_dm_delay(signal, dm, freq, n_points):
    #[1/s^2 * pc^-1*cm^3 ] * pc/cm^3 = 1/s^2
    # K_DM Constant for DM delay calculation
    delay = int(np.round(K_DM * dm / freq**2 * n_points))  # Delay in samples
    delay = (delay%n_points)/n_points
    return translate_signal(signal, delay)

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
            dispersed_pulse = apply_dm_delay(shifted_pulse, dms[j], freqs[j], n_points)

            shifted[i, j] = dispersed_pulse

    return original, shifted

def model_signal(*params, shifted_datasets):
    n_signals = shifted_datasets.shape[0]
    n_freqs = shifted_datasets.shape[1]
    n_points = shifted_datasets.shape[2]

    dms = params[:n_signals]
    shifts = params[n_signals:]

    modeled_signals = []

    for i in range(n_signals):  # model each signal i
        # Indices of other signals (to build template)
        other_indices = [j for j in range(n_signals) if j != i]

        # Build template across bands
        templates = []
        for f in range(n_freqs):
            # Undo DM + time shifts for all other signals
            aligned_signals = []
            for j in other_indices:
                undispersed = invert_dm_delay(
                    shifted_datasets[j, f], dms[j], freqs[f], n_points)
                aligned = translate_signal(undispersed, -shifts[j])
                aligned_signals.append(aligned)

            # Average template across other signals
            template = np.mean(aligned_signals, axis=0)
            templates.append(template)

        # Reapply DM and shift to simulate signal i
        band_signals = []
        for f, freq in enumerate(freqs):
            redispersed = apply_dm_delay(templates[f], dms[i], freq, n_points)
            shifted = translate_signal(redispersed, shifts[i])
            band_signals.append(shifted)

        modeled_signals.append(band_signals)

    return np.array(modeled_signals)  # shape: (n_signals, n_freqs, n_points)

# Funtion to plot model and signal
def plot_it(modeled_signals,datasets):
    x = np.linspace(0, 1, datasets.shape[2])
    n_pulses = datasets.shape[0]
    n_freqs = datasets.shape[1]
    offset_step = 1
    y_ticks = [i * offset_step for i in range(n_freqs)]
    y_labels = [f"{freq} MHz" for freq in freqs]
    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(n_pulses, 1, figsize=(12, 4 * n_pulses), sharex=True)
    for k in range(n_pulses):
        ax = axes[k] if n_pulses > 1 else axes  # handle 1 subplot edge case
        for band in range(n_freqs):
            offset = band * offset_step
            y_data_raw = datasets[k, band]  + offset
            y_model = modeled_signals[k, band]  + offset
            # Plot raw data in grey
            ax.plot(x, y_data_raw, color='grey', label=f"Data {freqs[band]} MHz" if k == 0 else "")
            # Plot model in dark blue
            ax.plot(x, y_model, color='darkblue', linestyle='--', label=f"Model {freqs[band]} MHz" if k == 0 else "")
            ax.set_ylabel(f"{freqs[band]} MHz")
        ax.set_title(f"Portrait {k+1}")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(f"Portrait {k + 1}")
        ax.set_ylabel("Frequency")
    # Common x-axis label at bottom
    axes[-1].set_xlabel("Phase")
    # Shared legend (optional)
    handles, labels = axes[0].get_legend_handles_labels()
    custom_lines = [
        Line2D([0], [0], color='grey', label='Data'),
        Line2D([0], [0], color='darkblue', linestyle='--', label='Template')
    ]
    fig.legend(handles=custom_lines, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Portraits", fontsize=16)
    plt.savefig(f"{plotdir}/model_fit.png")

# Generate the datasets (this is not shifted)
dms = np.random.normal(loc=base_dm, scale=variation_std, size=len(freqs)).tolist()
print("DM values:", [f"{dm:.6f}" for dm in dms])
shifts = np.random.uniform(low=-0.01, high=0.01, size=n_pulses).tolist()
print("Shift values:", [f"{s:.6f}" for s in shifts])

n_signals = len(shifts)
datasets, shifted_datasets = generate_signals(shifts, dms, freqs, n_points)

## Plot dataset
x = np.linspace(0, 1, datasets.shape[2])
n_pulses = datasets.shape[0]
n_freqs = datasets.shape[1]
offset_step = 1
y_ticks = [i * offset_step for i in range(n_freqs)]
y_labels = [f"{freq} MHz" for freq in freqs]
plt.figure(figsize=(12, 8))
fig, axes = plt.subplots(n_pulses, 1, figsize=(12, 4 * n_pulses), sharex=True)
for k in range(n_pulses):
    ax = axes[k] if n_pulses > 1 else axes  # handle 1 subplot edge case
    for band in range(n_freqs):
        offset = band * offset_step
        y_data = datasets[k, band]  + offset
        y_data_shifted = shifted_datasets[k, band]  + offset
        # Plot raw data in grey
        ax.plot(x, y_data, color='grey', label=f"Data {freqs[band]} MHz" if k == 0 else "")
        # Plot model in dark blue
        ax.plot(x, y_data_shifted, color='darkblue', linestyle='--', label=f"Model {freqs[band]} MHz" if k == 0 else "")
        ax.set_ylabel(f"{freqs[band]} MHz")
    ax.set_title(f"Portrait {k+1}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title(f"Portrait {k + 1}")
    ax.set_ylabel("Frequency")
# Common x-axis label at bottom
axes[-1].set_xlabel("Phase")
# Shared legend (optional)
handles, labels = axes[0].get_legend_handles_labels()
custom_lines = [
    Line2D([0], [0], color='grey', label='Unshifted Data'),
    Line2D([0], [0], color='darkblue', linestyle='--', label='Shifted Model')
]
fig.legend(handles=custom_lines, loc='upper right')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Portraits", fontsize=16)
plt.savefig(f"{plotdir}/data_shifted.png")

# Define the likelihood class
class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data, model, sigma, n_signals, parameter_prefixes=("dm", "d")):

        self.data = data
        self.model = model
        self.sigma = sigma
        self.N = data.shape[2]
        self.n_signals = n_signals
        self.freqs = freqs  # ensure freqs is accessible
        self.parameter_prefixes = parameter_prefixes

        # Build parameter dictionary
        parameters = {}
        for i in range(n_signals):
            parameters[f"{parameter_prefixes[0]}{i}"] = None
            parameters[f"{parameter_prefixes[1]}{i}"] = None

        super().__init__(parameters=parameters)

    def log_likelihood(self):
        # Extract parameters into ordered lists
        dms = [self.parameters[f"{self.parameter_prefixes[0]}{i}"] for i in range(self.n_signals)]
        shifts = [self.parameters[f"{self.parameter_prefixes[1]}{i}"] for i in range(self.n_signals)]

        # Generate model estimate
        est = self.model(*dms, *shifts, self.data)

        # Compute log likelihood
        ll = 0
        for k in range(self.n_signals):
            for band in range(len(self.freqs)):
                res = self.data[k, band] - est[k, band]
                ll += -0.5 * (
                    np.sum((res / self.sigma) ** 2) + self.N * np.log(2 * np.pi * self.sigma**2)
                )
        return ll

likelihood = SimpleGaussianLikelihood(data=shifted_datasets, 
                                      model=lambda *args: model_signal(*args, shifted_datasets=shifted_datasets), 
                                      n_signals=len(shifts), sigma=0.01)
    
# Set up and run the sampler
priors = {}
for i in range(n_signals):
    priors[f"dm{i}"] = bilby.core.prior.Uniform(9.5, 10.5, f"dm{i}")
    priors[f"d{i}"] = bilby.core.prior.Uniform(-0.3, 0.3, f"d{i}")

# Start timer
start_time = time.time()

sample = True
shutil.rmtree("outdir", ignore_errors=True)
# shuthill resets output directory so results aren’t mixed with old runs
# as sometimes resume=False does not work.
# It needs to be commented out if the intended is resume=True
if sample:
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=250,
        outdir=outdir,
        label=label,
        resume = False,
        clean=True,
    )
    result.plot_corner(save=False)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# Convert to h:m:s
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

print(f"\n⏱️ Elapsed time: {elapsed_time:.2f} seconds ({hours}h {minutes}m {seconds:.2f}s)")

# Load results
result = bilby.result.read_in_result(outdir=outdir, label=label)

# Use regex to get only dm* and d* parameters
dm_keys = [key for key in result.posterior.columns if re.match(r"^dm\d+$", key)]
d_keys = [key for key in result.posterior.columns if re.match(r"^d\d+$", key)]

# Sort keys to ensure dm0, dm1,... in order
dm_keys.sort()
d_keys.sort()

# Extract median values in order
dm_medians = [result.posterior[key].median() for key in dm_keys]
d_medians = [result.posterior[key].median() for key in d_keys]

# Combine into a single list of arguments
params = dm_medians + d_medians

# Evaluate model using best-fit parameters
modeled_signals = model_signal(*params, shifted_datasets=shifted_datasets)

# Print results for inspection
print("Best-fit DMs (median):", ", ".join(f"{dm:.2f}" for dm in dm_medians))
print("Best-fit Shifts (median):", ", ".join(f"{d:.2f}" for d in d_medians))

# Plot model vs data
plot_it(modeled_signals, shifted_datasets)