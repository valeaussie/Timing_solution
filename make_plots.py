import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

from utils import *

# Load results
result = bilby.result.read_in_result(outdir=outdir, label=label)

datasets = np.load(f"{outdir}/datasets.npy")
shifted_datasets = np.load(f"{outdir}/shifted_datasets.npy")

#print(datasets_1[0][0])
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
modeled_signals = model_signal(*params, shifted_datasets=shifted_datasets)

# Print results for inspection
print("Best-fit DMs (median):", ", ".join(f"{dm:.5f}" for dm in dm_medians))
print("Best-fit Shifts (median):", ", ".join(f"{d:.5f}" for d in d_medians))

# Plot the model and signal for each pulse
plot_it(modeled_signals, shifted_datasets)