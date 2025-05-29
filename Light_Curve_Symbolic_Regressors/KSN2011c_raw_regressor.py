import pandas as pd
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mark/Documents/Summer UROP 2025/One_Rising_Component_Curves/KSN2011c_raw_data.csv', sep=r'\s+')
df.columns = df.columns.str.strip()

t_exp = df[df["KJD-T_EXP"] > 0]["KJD"].min()
t_max = df[df["KJD-T_MAX"] > 0]["KJD"].min()
margin = 5  # Adjustable margin value
time_start = (t_exp - margin) - df["KJD"].min()
time_end = (t_max + margin) - df["KJD"].min()

df['time'] = df['KJD'] - df['KJD'].min()
df_model = df[(df['time'] >= time_start) & (df['time'] <= time_end)]

x_data = df_model['time'].values
y_data = df_model['LC_val-BCK'].values
y_error = df_model['LC_err'].values

y_noisy = y_data

X = x_data.reshape(-1, 1)
y = y_noisy

model = PySRRegressor(
    niterations=100,
    populations=24,
    ncycles_per_iteration=100,
    binary_operators=["+", "-", "*", "/", "^",],
    unary_operators=["sin", "log", "erf"],
    constraints={"^": (0, 1)},
    nested_constraints={"sin": {"sin": 0}},
    loss="L2DistLoss()",
    model_selection="best",
    maxsize=50,
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
        "erf": 5,
    }
)

model.fit(X, y)

eval_start = t_exp - df['KJD'].min()
eval_end = t_max - df['KJD'].min()
eval_mask = (x_data >= eval_start) & (x_data <= eval_end)
x_eval = x_data[eval_mask]
y_eval = y_data[eval_mask]
y_error_eval = y_error[eval_mask]

y_pred_eval = model.predict(x_eval.reshape(-1, 1))
ss_res = np.sum((y_eval - y_pred_eval) ** 2)
ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
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

x_smooth = np.linspace(eval_start, eval_end, 10000).reshape(-1, 1)
y_smooth_pred = model.predict(x_smooth)

plt.figure(figsize=(8, 5))
plt.errorbar(x_eval, y_eval, yerr=y_error_eval, fmt='o', color='black', alpha=0.1, label="Observed Data", capsize=3)
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