import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time
import re
import shutil

plt.close('all')

from utils import *


# Generate the datasets (this is not shifted)
dms = np.random.normal(loc=base_dm, scale=variation_std, size=n_pulses).tolist()
print("DM values:", [f"{dm:.6f}" for dm in dms])
shifts = np.random.uniform(low=-0.01, high=0.01, size=n_pulses).tolist()
print("Shift values:", [f"{s:.6f}" for s in shifts])

n_signals = len(shifts)
datasets, shifted_datasets = generate_signals(shifts, dms, freqs, n_points)

# ## Plot dataset
# x = np.linspace(0, 1, datasets.shape[2])
# n_pulses = datasets.shape[0]
# n_freqs = datasets.shape[1]
# offset_step = 1
# y_ticks = [i * offset_step for i in range(n_freqs)]
# y_labels = [f"{freq} MHz" for freq in freqs]
# plt.figure(figsize=(12, 8))
# fig, axes = plt.subplots(n_pulses, 1, figsize=(12, 4 * n_pulses), sharex=True)
# for k in range(n_pulses):
#     ax = axes[k] if n_pulses > 1 else axes  # handle 1 subplot edge case
#     for band in range(n_freqs):
#         offset = band * offset_step
#         y_data = datasets[k, band]  + offset
#         y_data_shifted = shifted_datasets[k, band]  + offset
#         # Plot raw data in grey
#         ax.plot(x, y_data, color='grey', label=f"Data {freqs[band]} MHz" if k == 0 else "")
#         # Plot model in dark blue
#         ax.plot(x, y_data_shifted, color='darkblue', linestyle='--', label=f"Model {freqs[band]} MHz" if k == 0 else "")
#         ax.set_ylabel(f"{freqs[band]} MHz")
#     ax.set_title(f"Portrait {k+1}")
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_labels)
#     ax.set_title(f"Portrait {k + 1}")
#     ax.set_ylabel("Frequency")
# # Common x-axis label at bottom
# axes[-1].set_xlabel("Phase")
# # Shared legend (optional)
# handles, labels = axes[0].get_legend_handles_labels()
# custom_lines = [
#     Line2D([0], [0], color='grey', label='Unshifted Data'),
#     Line2D([0], [0], color='darkblue', linestyle='--', label='Shifted Data')
# ]
# fig.legend(handles=custom_lines, loc='upper right')
# fig.tight_layout(rect=[0, 0, 1, 0.96])
# fig.suptitle("Portraits", fontsize=16)
# plt.savefig(f"{plotdir}/data_shifted.png")

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
                                     model=lambda *args: model_signal(*args, shifted_datasets=shifted_datasets)[1], 
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

print(f"\n Elapsed time: {elapsed_time:.2f} seconds ({hours}h {minutes}m {seconds:.2f}s)")

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

# Print dm_medians and d_medians
print("Best-fit DMs (median):", ", ".join(f"{dm:.5f}" for dm in dm_medians))
print("Best-fit Shifts (median):", ", ".join(f"{d:.5f}" for d in d_medians))

np.save(f"{outdir}/datasets.npy", datasets)
np.save(f"{outdir}/shifted_datasets.npy", shifted_datasets)