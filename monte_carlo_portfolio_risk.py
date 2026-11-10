# Monte Carlo Portfolio Risk: VaR, Expected Shortfall, and the correlation lesson
# Colab: paste into one cell (or split at the "# %%" markers) and run.
# If yfinance is missing: !pip install -q yfinance

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# ============================ CONFIG ============================
PORTFOLIOS = {
    "A: Concentrated Tech": {"AAPL": 0.40, "MSFT": 0.30, "NVDA": 0.30},
    "B: Diversified":       {"MSFT": 0.40, "JNJ": 0.30, "PG": 0.30},
}
START   = "2015-01-01"
HORIZON = 21          # trading days (~1 month)
N_SIMS  = 10_000
CONF    = [0.95, 0.99]
T_DF    = 5           # Student-t degrees of freedom (fat tails)
VALUE   = 100_000     # portfolio value for $ figures
SEED    = 42
OUT     = "outputs"
# Drift is set to 0 on purpose: historical mean returns are noise at a 1-month horizon.

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


# %% ============================ CORE FUNCTIONS ============================
def load_prices(tickers):
    px = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    return px[tickers].dropna()


def draw(n, k):
    """Random draws, reused across models so Normal vs t is apples to apples."""
    Z = rng.standard_normal((n, k))
    W = rng.chisquare(T_DF, size=(n, 1)) / T_DF
    return Z, W


def simulate(cov_m, Z, W, kind):
    """Correlated 1-month log returns. 't' = multivariate Student-t scaled to the same covariance."""
    L = np.linalg.cholesky(cov_m)
    X = Z if kind == "normal" else Z / np.sqrt(W) * np.sqrt((T_DF - 2) / T_DF)
    return X @ L.T


def metrics(port_ret):
    """VaR and ES as positive loss fractions for each confidence level."""
    loss = -port_ret
    out = {}
    for c in CONF:
        v = np.quantile(loss, c)
        out[c] = (v, loss[loss >= v].mean())
    return out


def corr_sweep(vols, Z, W, w, rhos):
    """Hold each stock's volatility fixed; change only a uniform pairwise correlation."""
    k = len(vols)
    res = {"var95": [], "es95": [], "es95_t": []}
    for rho in rhos:
        C = np.full((k, k), rho)
        np.fill_diagonal(C, 1.0)
        cov = np.outer(vols, vols) * C
        for kind, key_v, key_e in [("normal", "var95", "es95"), ("t", None, "es95_t")]:
            port = (np.exp(simulate(cov, Z, W, kind)) - 1) @ w
            v, e = metrics(port)[0.95]
            if key_v:
                res[key_v].append(v)
            res[key_e].append(e)
    return {k_: np.array(v) for k_, v in res.items()}


def backtest(px, w, window=252):
    """Walk-forward: estimate covariance on past year, forecast VaR, compare to next realized month."""
    lr = np.log(px).diff().dropna()
    n = len(lr)
    breaches = {(m, c): 0 for m in ["normal", "t"] for c in CONF}
    months = 0
    for t in range(window, n - HORIZON, HORIZON):
        cov_m = lr.iloc[t - window:t].cov().values * HORIZON
        sig = np.sqrt(w @ cov_m @ w)
        real = w @ (np.exp(lr.iloc[t:t + HORIZON].sum().values) - 1)
        for c in CONF:
            var_n = stats.norm.ppf(c) * sig
            var_t = stats.t.ppf(c, T_DF) * np.sqrt((T_DF - 2) / T_DF) * sig
            breaches[("normal", c)] += real < -var_n
            breaches[("t", c)] += real < -var_t
        months += 1
    return months, {k_: v / months for k_, v in breaches.items()}


def simulate_paths(cov_m, w, n):
    """Daily buy-and-hold paths over the horizon (normal model). Returns (HORIZON+1, n) value index, start = 1.
    Uses its own generator so the main results above stay unchanged."""
    g = np.random.default_rng(SEED + 1)
    L = np.linalg.cholesky(cov_m / HORIZON)
    cum = np.cumsum(g.standard_normal((HORIZON, n, len(w))) @ L.T, axis=0)
    return np.vstack([np.ones(n), (np.exp(cum) * w).sum(axis=2)])


# %% ============================ RUN ============================
store, rows = {}, []

for name, wd in PORTFOLIOS.items():
    tickers = list(wd)
    w = np.array([wd[t] for t in tickers])
    px = load_prices(tickers)
    lr = np.log(px).diff().dropna()
    cov_m = lr.cov().values * HORIZON
    corr = lr.corr().values
    k = len(tickers)
    avg_rho = corr[np.triu_indices(k, 1)].mean()

    Z, W = draw(N_SIMS, k)
    sims = {}
    for kind in ["normal", "t"]:
        R = simulate(cov_m, Z, W, kind)
        stock_ret = np.exp(R) - 1
        port = stock_ret @ w
        sims[kind] = (stock_ret, port)
        m = metrics(port)
        rows.append({
            "Portfolio": name, "Model": kind,
            "VaR95 %": m[0.95][0] * 100, "ES95 %": m[0.95][1] * 100,
            "VaR99 %": m[0.99][0] * 100, "ES99 %": m[0.99][1] * 100,
            "VaR95 $": m[0.95][0] * VALUE, "ES95 $": m[0.95][1] * VALUE,
        })

    # Standalone vs portfolio VaR (normal model): the diversification benefit
    stock_ret, port = sims["normal"]
    standalone = np.array([np.quantile(-stock_ret[:, i], 0.95) for i in range(k)])
    port_var = metrics(port)[0.95][0]
    div_benefit = 1 - port_var / (w @ standalone)

    # Parametric sanity check (zero drift, normal)
    sigma_p = np.sqrt(w @ cov_m @ w)
    param_var = stats.norm.ppf(0.95) * sigma_p

    # Correlation sweep
    rhos = np.linspace(0, 0.95, 20)
    sweep = corr_sweep(np.sqrt(np.diag(cov_m)), Z, W, w, rhos)

    # Backtest
    months, bt = backtest(px, w)

    # Daily paths for the fan chart
    paths = simulate_paths(cov_m, w, N_SIMS)

    store[name] = dict(tickers=tickers, w=w, corr=corr, avg_rho=avg_rho, sims=sims,
                       standalone=standalone, div_benefit=div_benefit,
                       param_var=param_var, mc_var=port_var,
                       rhos=rhos, sweep=sweep, months=months, bt=bt, paths=paths,
                       ann_vol=lr.std().values * np.sqrt(252))

    pd.DataFrame({"normal": sims["normal"][1], "student_t": sims["t"][1]}) \
        .to_csv(f"{OUT}/simulation_{name[0].lower()}.csv", index=False)

summary = pd.DataFrame(rows).round(2)
print(summary.to_string(index=False))
summary.to_csv(f"{OUT}/summary.csv", index=False)

# Probability table: chance of each 1-month outcome
prob_rows = []
for name in store:
    for kind in ["normal", "t"]:
        port = store[name]["sims"][kind][1]
        prob_rows.append({
            "Portfolio": name, "Model": kind,
            "P(profit) %": (port > 0).mean() * 100,
            "P(loss>5%) %": (port < -0.05).mean() * 100,
            "P(loss>10%) %": (port < -0.10).mean() * 100,
            "P(loss>20%) %": (port < -0.20).mean() * 100,
            "P(gain>5%) %": (port > 0.05).mean() * 100,
            "P(gain>10%) %": (port > 0.10).mean() * 100,
            "P(gain>20%) %": (port > 0.20).mean() * 100,
        })
probs = pd.DataFrame(prob_rows).round(1)
print("\nPROBABILITY TABLE (1-month, zero drift)")
print(probs.to_string(index=False))
probs.to_csv(f"{OUT}/probability_table.csv", index=False)


# %% ============================ CHARTS ============================
names = list(store)

# 1) Loss histogram with VaR / ES lines (one per portfolio)
for name in names:
    s = store[name]
    port = s["sims"]["normal"][1] * 100
    m = metrics(s["sims"]["normal"][1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(port, bins=80, color="#4C78A8", alpha=0.85)
    ax.axvline(-m[0.95][0] * 100, color="#F58518", lw=2, label=f"95% VaR: -{m[0.95][0]*100:.1f}%")
    ax.axvline(-m[0.99][0] * 100, color="#E45756", lw=2, label=f"99% VaR: -{m[0.99][0]*100:.1f}%")
    ax.axvline(-m[0.95][1] * 100, color="black", lw=2, ls="--", label=f"95% Exp. Shortfall: -{m[0.95][1]*100:.1f}%")
    ax.set_title(f"{name}: 10,000 simulated 1-month returns")
    ax.set_xlabel("Portfolio return (%)")
    ax.set_ylabel("Scenarios")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hist_{name[0].lower()}.png")
    plt.show()

# 2) Correlation heatmaps side by side
fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.5))
for ax, name in zip(np.atleast_1d(axes), names):
    s = store[name]
    im = ax.imshow(s["corr"], vmin=0, vmax=1, cmap="Reds")
    ax.set_xticks(range(len(s["tickers"])), s["tickers"])
    ax.set_yticks(range(len(s["tickers"])), s["tickers"])
    for i in range(len(s["tickers"])):
        for j in range(len(s["tickers"])):
            ax.text(j, i, f"{s['corr'][i, j]:.2f}", ha="center", va="center",
                    color="white" if s["corr"][i, j] > 0.6 else "black")
    ax.set_title(f"{name}\navg correlation {s['avg_rho']:.2f}")
fig.tight_layout()
fig.savefig(f"{OUT}/correlation_heatmaps.png")
plt.show()

# 3) KEY CHART: VaR / ES vs correlation, volatility held constant
for name in names:
    s = store[name]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s["rhos"], s["sweep"]["var95"] * 100, color="#F58518", lw=2.5, label="95% VaR")
    ax.plot(s["rhos"], s["sweep"]["es95"] * 100, color="black", lw=2.5, label="95% Expected Shortfall")
    ax.plot(s["rhos"], s["sweep"]["es95_t"] * 100, color="black", lw=2, ls="--", label="95% ES (fat tails, Student-t)")
    ax.axvline(s["avg_rho"], color="grey", ls=":", label=f"Actual avg correlation ({s['avg_rho']:.2f})")
    ax.set_title(f"{name}: same stock volatilities, only correlation changes")
    ax.set_xlabel("Pairwise correlation (rho)")
    ax.set_ylabel("1-month loss (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/var_vs_correlation_{name[0].lower()}.png")
    plt.show()

# 4) Portfolio comparison (VaR95 and ES95, Normal vs Student-t)
cmp_df = summary.set_index(["Portfolio", "Model"])
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(names))
bw = 0.2
for i, (model, metric, col) in enumerate([("normal", "VaR95 %", "#F58518"), ("normal", "ES95 %", "#E45756"),
                                          ("t", "VaR95 %", "#72B7B2"), ("t", "ES95 %", "#4C78A8")]):
    vals = [cmp_df.loc[(n, model), metric] for n in names]
    ax.bar(x + (i - 1.5) * bw, vals, bw, label=f"{metric.split()[0]} ({'Normal' if model == 'normal' else 'Student-t'})", color=col)
ax.set_xticks(x, [n.split(":")[0] + ": " + n.split(": ")[1] for n in names])
ax.set_ylabel("1-month loss (%)")
ax.set_title("Risk by portfolio and model")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/portfolio_comparison.png")
plt.show()


# 5) Fan chart: percentile bands of simulated daily paths
for name in names:
    s_ = store[name]
    v = s_["paths"] * VALUE
    days = np.arange(HORIZON + 1)
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95], axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in rng.choice(N_SIMS, 40, replace=False):
        ax.plot(days, v[:, i], color="grey", alpha=0.15, lw=0.6)
    ax.fill_between(days, p5, p95, color="#4C78A8", alpha=0.25, label="5th-95th percentile")
    ax.fill_between(days, p25, p75, color="#4C78A8", alpha=0.45, label="25th-75th percentile")
    ax.plot(days, p50, color="black", lw=2, label="Median")
    ax.axhline(VALUE, color="#F58518", ls="--", lw=1.5, label="Start")
    ax.set_title(f"{name}: simulated portfolio value over 1 month")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fan_{name[0].lower()}.png")
    plt.show()


# %% ============================ SLIDE NUMBERS ============================
print("\n===== NUMBERS FOR YOUR SLIDES =====")
for name in names:
    s = store[name]
    sw = s["sweep"]
    mn, mt = metrics(s["sims"]["normal"][1]), metrics(s["sims"]["t"][1])
    print(f"\n{name}  ({', '.join(s['tickers'])})")
    print(f"  Annualized vol:         {', '.join(f'{t} {v*100:.0f}%' for t, v in zip(s['tickers'], s['ann_vol']))}")
    print(f"  Avg pairwise corr:      {s['avg_rho']:.2f}")
    print(f"  95% VaR / ES (normal):  {mn[0.95][0]*100:.1f}% / {mn[0.95][1]*100:.1f}%   (${mn[0.95][0]*VALUE:,.0f} / ${mn[0.95][1]*VALUE:,.0f} on ${VALUE:,})")
    print(f"  99% VaR / ES (normal):  {mn[0.99][0]*100:.1f}% / {mn[0.99][1]*100:.1f}%")
    pr = probs[(probs["Portfolio"] == name) & (probs["Model"] == "normal")].iloc[0]
    print(f"  P(loss>10%) / P(loss>20%) / P(profit): {pr['P(loss>10%) %']:.1f}% / {pr['P(loss>20%) %']:.1f}% / {pr['P(profit) %']:.1f}%")
    print(f"  ES95 normal -> t:       {mn[0.95][1]*100:.1f}% -> {mt[0.95][1]*100:.1f}%")
    print(f"  VaR95 at rho 0 -> 0.95: {sw['var95'][0]*100:.1f}% -> {sw['var95'][-1]*100:.1f}%  (x{sw['var95'][-1]/sw['var95'][0]:.2f})")
    print(f"  Diversification benefit (portfolio VaR vs weighted standalone VaR): {s['div_benefit']*100:.0f}%")
    print(f"  Sanity: MC VaR95 {s['mc_var']*100:.2f}% vs parametric {s['param_var']*100:.2f}%")
    b = s["bt"]
    print(f"  Backtest ({s['months']} months): 95% breach rate normal {b[('normal',0.95)]*100:.1f}% / t {b[('t',0.95)]*100:.1f}% (expect 5%); "
          f"99% breach rate normal {b[('normal',0.99)]*100:.1f}% / t {b[('t',0.99)]*100:.1f}% (expect 1%)")

a, b_ = names[0], names[1]
va = metrics(store[a]["sims"]["normal"][1])[0.95][0]
vb = metrics(store[b_]["sims"]["normal"][1])[0.95][0]
print(f"\nPortfolio B VaR95 is {(1 - vb/va)*100:.0f}% lower than Portfolio A.")

# Optional: zip and download all outputs in Colab
try:
    import shutil
    from google.colab import files
    shutil.make_archive("outputs", "zip", OUT)
    files.download("outputs.zip")
except Exception:
    pass
