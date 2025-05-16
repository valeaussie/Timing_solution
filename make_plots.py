import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import os

from utils import *

# Load results
result = bilby.result.read_in_result(outdir=outdir, label=label)

datasets = np.load(f"{outdir}/datasets.npy")
shifted_datasets = np.load(f"{outdir}/shifted_datasets.npy")

#print(datasets[0][0])
print(np.shape(datasets))
#print(datasets[0][0])

# Load results
result = bilby.result.read_in_result(outdir=outdir, label=label)

# Get only dm* and d* parameters and sort
dm_keys = [key for key in result.posterior.columns if re.match(r"^dm\d+$", key)]
d_keys = [key for key in result.posterior.columns if re.match(r"^d\d+$", key)]
dm_keys.sort()
d_keys.sort()

# Extract median values in order
dm_medians = [result.posterior[key].median() for key in dm_keys]
d_medians = [result.posterior[key].median() for key in d_keys]

# Combine into a single list of arguments
params = dm_medians + d_medians
print("the numbers of parameters is: ", len(params))

# Evaluate model using best-fit parameters
template, modeled_signals = model_signal(*params, shifted_datasets=shifted_datasets)

print("the shape of the template is: ", np.shape(template))

# Print results for inspection
print("Best-fit DMs (median):", ", ".join(f"{dm:.5f}" for dm in dm_medians))
print("Best-fit Shifts (median):", ", ".join(f"{d:.5f}" for d in d_medians))

# Plot the model and signal for each pulse
#plot_it(modeled_signals, shifted_datasets)

x = np.linspace(0, 1, n_points)
for i in range(n_pulses):
    # Plot average profile
    mean_port_shifted = np.mean(shifted_datasets, axis=1)
    mean_template = np.mean(template, axis=1)
    plt.plot(x, mean_template[i], label='Template')
    plt.plot(x, mean_port_shifted[i], label='Shifted Data')
    plt.title('Mean Profile')
    plt.legend()
    plt.xlabel('Phase')
    plt.savefig(f"{plotdir}/mean_profile_shifted_data_{i}.png")

    plt.close()

# Plot average profile
for i in range(n_pulses):
    # Plot average profile
    mean_port = np.mean(datasets, axis=1)
    mean_template = np.mean(template, axis=1)
    plt.plot(x, mean_template[i], label='Template')
    plt.plot(x, mean_port[i], label='Original Data')
    plt.title('Mean Profile')
    plt.legend()
    plt.xlabel('Phase')
    plt.savefig(f"{plotdir}/mean_profile_data_{i}.png")

    plt.close()

# Plot templet vs model for each frequency band
mean=0.5, 
sigma=0.05
x = np.linspace(0, 1, n_points)
gaussian = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
for i in range(len(freqs)):
    mean_template = np.mean(template, axis=0)
    plt.plot(x, gaussian, label=f'Model')
    plt.plot(x, mean_template[i], label=f'Reconstructed template')
    plt.title(f'Mean Profile for freq {i}')
    plt.legend(
        loc='upper right',            # Position
        bbox_to_anchor=(1.0, 1.0),    # Anchor outside plot
        labelspacing=1.2,             # Line spacing
        borderaxespad=0.5,            # Padding around legend
        frameon=True                  # Optional: draw box around legend
    )
    plt.xlabel('Phase')
    plt.savefig(f"{plotdir}/template_vs_model{i}.png")

    plt.close()



for i in range(n_pulses):
    mean_template = np.mean(template, axis=1)
    flux_map = template[i]
    # Set up figure with two vertical subplots
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)  # Top smaller than bottom

    # Top: Mean Profile
    ax0 = plt.subplot(gs[0])
    ax0.plot(x, mean_template[i], color='black')
    ax0.set_xticks([])
    ax0.set_xlim(0, 1)
    #ax0.set_xlim(x.min(), x.max())
    ax0.set_title(f'Average pulse profile and dynamic spectrum pulse {i}')
    ax0.tick_params(labelbottom=False)
    ax0.tick_params(labelleft=False)

    # Bottom: Flux Map
    ax1 = plt.subplot(gs[1], sharex=ax0)
    im = ax1.imshow(flux_map, aspect='auto', cmap='hot', origin='lower',
                    extent=[0, 1, freqs[0], freqs[-1]])
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Freq [MHz]')
    ax1.set_yticks(freqs)
    phase_ticks = np.linspace(0, 1, 6)
    ax1.set_xticks(phase_ticks)
    ax1.set_xlim(0, 1)

    plt.savefig(f"{plotdir}/combined_pulse_flux_map_{i}_dedispersed.png", dpi=300)

for i in range(n_pulses):
    mean_shifted_datasets = np.mean(shifted_datasets, axis=1)
    flux_map = shifted_datasets[i]
    # Set up figure with two vertical subplots
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)  # Top smaller than bottom

    # Top: Mean Profile
    ax0 = plt.subplot(gs[0])
    ax0.plot(x, mean_shifted_datasets[i], color='black')
    ax0.set_xticks([])
    ax0.set_xlim(0, 1)
    #ax0.set_xlim(x.min(), x.max())
    ax0.set_title(f'Average pulse profile and dynamic spectrum pulse {i}')
    ax0.tick_params(labelbottom=False)
    ax0.tick_params(labelleft=False)

    # Bottom: Flux Map
    ax1 = plt.subplot(gs[1], sharex=ax0)
    im = ax1.imshow(flux_map, aspect='auto', cmap='hot', origin='lower',
                    extent=[0, 1, freqs[0], freqs[-1]])
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Freq [MHz]')
    ax1.set_yticks(freqs)
    phase_ticks = np.linspace(0, 1, 6)
    ax1.set_xticks(phase_ticks)
    ax1.set_xlim(0, 1)

    plt.savefig(f"{plotdir}/combined_pulse_flux_map_{i}_dispersed.png", dpi=300)