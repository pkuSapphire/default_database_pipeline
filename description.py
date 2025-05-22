# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline import get_final_dataframe

# Load data using pipeline
df = get_final_dataframe()
df["at_billion"] = df["at"] / 1e3

# Define bins and labels
bins = [0, 0.5, 1, 2, 5, 10, np.inf]
labels = ["<0.5B", "0.5–1B", "1–2B", "2–5B", "5–10B", ">10B"]
df["size_bin"] = pd.cut(df["at_billion"], bins=bins, labels=labels)

# Group and calculate totals
size_template = df.groupby("size_bin").agg(
    total_firms=("gvkey", "count"),
    total_defaults=("dflt_flag", "sum")
).reset_index()

size_template["statements_pct"] = size_template["total_firms"] / size_template["total_firms"].sum()
size_template["defaults_pct"] = size_template["total_defaults"] / df["dflt_flag"].sum()

# Plot the corrected figure
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(size_template["size_bin"], size_template["statements_pct"], color="navy", label="Statements")
ax.plot(size_template["size_bin"], size_template["defaults_pct"], "rD", label="Defaults")

ax.set_ylabel("Percentage")
ax.set_yticks(np.arange(0, 0.46, 0.05))
ax.set_yticklabels([f"{int(x*100)}%" for x in np.arange(0, 0.46, 0.05)])
ax.set_title("Figure 3: Distribution of Statements and Defaults by Size")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()