import pandas as pd
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mark/Documents/Summer UROP 2025/Two_Rising_Component_Curves/SN2018oh_binned_data.csv', sep=r'\s+')

df['time'] = df['time'] - df['time'].min()

x_temp = df['time'].values
y_temp = df['flux'].values

T_exp = None
for i in range(5, len(y_temp)):
    flux_subset = y_temp[:i+1]
    mean_flux = np.mean(flux_subset)
    std_flux = np.std(flux_subset)
    if y_temp[i] > mean_flux + 5 * std_flux:
        T_exp = x_temp[i]
        break

T_max = x_temp[np.argmax(y_temp)]

margin = 2.0

df_fit = df[(df['time'] >= T_exp - margin) & (df['time'] <= T_max + margin)]
x_fit = df_fit['time'].values
y_fit = df_fit['flux'].values
y_fit_error = df_fit['flux_err'].values

df_plot = df[(df['time'] >= T_exp) & (df['time'] <= T_max)]
x_data = df_plot['time'].values
y_data = df_plot['flux'].values
y_error = df_plot['flux_err'].values
y_noisy = y_data

X = x_fit.reshape(-1, 1)
y = y_fit

model = PySRRegressor(
    niterations=1000,
    populations=24,
    ncycles_per_iteration=100,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "log", "erf"],
    constraints={"^": (0, 1)},
    nested_constraints={"sin": {"sin": 0}},
    elementwise_loss="L2DistLoss()",
    model_selection="best",
    maxsize=30,
    parsimony=0.0001,
    weight_optimize=0.001,
    turbo=True,
    complexity_of_operators={
        "+": 1,
        "-": 1,
        "*": 1,
        "/": 2,
        "^": 2,
        "log": 3,
        "sin": 3,
        "erf": 3,
    }
)

model.fit(X, y)

y_pred_eval = model.predict(x_data.reshape(-1, 1))
ss_res = np.sum((y_data - y_pred_eval) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r2 = 1 - (ss_res / ss_tot)

best_model = model.get_best()
best_equation = best_model['equation']
complexity = best_model['complexity']
loss = best_model['loss']
score = best_model['score']

print("\n=== PySR Best-Fit Equation ===")
print(f"Equation: {best_equation}")
print(f"R^2 Score: {r2:.4f}")
print(f"Complexity: {complexity}")
print(f"Loss: {loss:.4f}")
print(f"Score: {score:.4f}")

x_smooth = np.linspace(min(x_data), max(x_data), 10000).reshape(-1, 1)
y_smooth_pred = model.predict(x_smooth)

plt.figure(figsize=(8, 5))
plt.errorbar(x_data, y_data, yerr=y_error, fmt='o', color='black', alpha=0.1, label="Observed Data", capsize=3)
plt.plot(x_smooth, y_smooth_pred, label="PySR Best-Fit Curve", color="purple", linewidth=2)

plt.title("PySR Symbolic Regression Fit to Light Curve")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.legend()
plt.grid(True)

plt.text(
    0.95, 0.05,
    f"Curve of Best Fit = {best_equation}\nR^2 = {r2:.4f}\nComplexity = {complexity}\nLoss = {loss:.4f}\nScore = {score:.4f}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.show()