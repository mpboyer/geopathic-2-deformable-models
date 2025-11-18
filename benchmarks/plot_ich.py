import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import scipy.stats as stats
import numpy as np

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

data = pd.read_csv('benchmarks/csv/ich_benchmark_11:18_13:07:24.csv')

plot = sns.relplot(
    height=4,
    aspect=1.2,
    data=data,
    x="vertices",
    y="time",
    hue="model",
    kind="line",
    marker="o"
)

plt.xlabel("Number of Vertices")
plt.ylabel("Time")
plt.xscale("log")
plt.yscale("log")

x = np.log10(data["vertices"])
y = np.log10(data["time"])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept
plot.ax.plot(10**x_fit, 10**y_fit, color="tab:blue", label="Linear fit (log-log)")
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

data["source"] = data["source"].astype(str)
data["model"] = data["model"].astype(str)

# Annotate each (source, model) group at the average (vertices, time) position
for model, group in data.groupby(["model"]):
    avg_vertices = group["vertices"].mean()
    avg_time = group["time"].mean()
    # Offset label to the right by 5% of the average x value
    offset_vertices = avg_vertices * 1.15
    plot.ax.text(
        offset_vertices, avg_time, str(model[0][:-4]),
        fontsize=8, ha='left', va='center'
    )

plot.figure.tight_layout()
plot.legend.remove()
plot.savefig("benchmarks/pdf/ich_benchmark.pdf", bbox_inches="tight")