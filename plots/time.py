import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

sns.set_context("paper")

figures_dir = "./plots/time.png"
data_path = Path("./plots/data_for_plots")

hmc = np.load(data_path / Path(f"hmc_time_German.npy"))
learned_kernel = np.load(data_path / Path(f"learned_kernel_time_German.npy"))

data_hmc = {
    'x': hmc[0],
    'y': hmc[1],
    }

data_learned_kernel = {
    'x': learned_kernel[:13],
    'y1': learned_kernel[13:],
    }


df_hmc = pd.DataFrame(data_hmc)
df_learned_kernel = pd.DataFrame(data_learned_kernel)

# palette='viridis',

sns.set(style="whitegrid", rc={"grid.alpha": 1.})

sns.lineplot(x='x', y='y', data=df_hmc, alpha=1., label='HMC', linewidth=5)
sns.lineplot(x='x', y='y1', data=df_learned_kernel, alpha=1., label='Learned kernel', linewidth=5)

# plt.title('Your Title')

plt.xlabel('Parallel chains')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin as needed

plt.savefig(figures_dir)
