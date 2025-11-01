import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
paper_bg   = "#f0f2f5"
grid_color = "#ffffff"
opt_color  = "#f2a900"  # Optimistic (orange)
pes_color  = "#d14b4b"  # Pessimistic (red)

plt.rcParams.update({
    "figure.figsize": (9.5, 6),
    "axes.facecolor": paper_bg,
    "axes.edgecolor": "none",
    "axes.grid": True,
    "grid.color": grid_color,
    "grid.linewidth": 1.2,
    "grid.alpha": 1.0,
    "font.size": 16,
})

def ema(x, alpha=0.1):
    y, m = np.empty_like(x, dtype=float), 0.0
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m if i else v
        y[i] = m
    return y

df_opt = pd.read_csv("scalars_export-v2-beta=0.csv")        # optimistic β=0
df_pes = pd.read_csv("scalars_export-v2-beta=0_1_test.csv") # pessimistic β=-1

def get_reward_curve(df):
    r = (df[df.tag == "Reward/Test"][["step","value"]]
           .rename(columns={"value":"reward"})
           .sort_values("step")
           .drop_duplicates("step", keep="last"))
    return r["step"].to_numpy(), r["reward"].to_numpy()

x_opt, y_opt = get_reward_curve(df_opt)
x_pes, y_pes = get_reward_curve(df_pes)

# Smooth (EMA or rolling—EMA is robust when steps are not perfectly uniform)
y_opt_s = ema(y_opt, alpha=0.1)
y_pes_s = ema(y_pes, alpha=0.1)

plt.figure()
plt.plot(x_pes, y_pes_s, label="Pessimistic", color=pes_color, linewidth=3)
plt.plot(x_opt, y_opt_s, label="Optimistic", color=opt_color, linewidth=3)
plt.xlabel("Time steps (1e6)")
plt.ylabel("Reward")
plt.title("TOP-TD3 under fixed beta | HalfCheetah-v2 | seed 1")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()
