# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# my personal black-scholes folder has a hyphen in its name, so it can't be imported as a package;
# add it to the path so we can reuse the functions from bsm_formula.py as our benchmark
sys.path.append(str(Path(__file__).resolve().parents[1] / "black-scholes"))
from bsm_formula import calc_d1_d2, bsm_call, bsm_put, put_call_parity # type: ignore


# function to calculate the CRR tree parameters (u, d, p) and the one-step discount factor
def crr_parameters(t, r, q, sigma, steps) -> tuple[float, float, float, float]:
    """
    ### args:
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate
    - sigma: volatility
    - steps: number of steps (N) in the tree

    ### returns:
    u (up factor), d (down factor), p (risk-neutral up probability), disc (one-step discount factor)
    """

    dt: float = t / steps
    u: float = np.exp(sigma * np.sqrt(dt))
    d: float = 1 / u  # recombining tree: an up then a down lands back where you started
    p: float = (np.exp((r - q) * dt) - d) / (u - d)  # makes the stock grow at (r - q) on average
    disc: float = np.exp(-r * dt)

    return u, d, p, disc


# payoff at exercise for a call or put (works on a single price or a numpy array of prices)
def payoff(prices, X, option_type):
    if option_type == "call":
        return np.maximum(prices - X, 0.0)
    if option_type == "put":
        return np.maximum(X - prices, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


# function to price an option with a CRR binomial tree (backward induction)
def crr_price(S, X, t, r, q, sigma, steps, option_type="call", american=False) -> float:
    """
    ### args:
    - S: Underlying stock price
    - X: Exercise Price
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate
    - sigma: volatility
    - steps: number of steps (N) in the tree
    - option_type: "call" or "put"
    - american: if True, allow early exercise at every node

    ### returns:
    the value of the option at the first node of the tree
    """

    u, d, p, disc = crr_parameters(t, r, q, sigma, steps)

    # terminal stock prices: j up-moves and (steps - j) down-moves
    j = np.arange(steps + 1)
    values = payoff(S * u**j * d ** (steps - j), X, option_type)

    # walk backward one step at a time; each value is the discounted expectation of its two children
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])  # [1:] = up child, [:-1] = down child

        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, payoff(S * u**j * d ** (i - j), X, option_type))

    return float(values[0])


# function that keeps the whole tree so it can be printed (use small step counts)
def crr_tree(S, X, t, r, q, sigma, steps, option_type="call", american=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ### returns:
    two DataFrames (stock prices, option values). Columns are time steps 0..N,
    rows are the number of up-moves j. Cells that can't exist are NaN.
    """

    u, d, p, disc = crr_parameters(t, r, q, sigma, steps)

    stock = np.full((steps + 1, steps + 1), np.nan)
    option = np.full((steps + 1, steps + 1), np.nan)

    for i in range(steps + 1):
        j = np.arange(i + 1)
        stock[j, i] = S * u**j * d ** (i - j)

    option[:, steps] = payoff(stock[:, steps], X, option_type)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option[j, i] = disc * (p * option[j + 1, i + 1] + (1 - p) * option[j, i + 1])
            if american:
                option[j, i] = max(option[j, i], payoff(stock[j, i], X, option_type))

    cols = [f"step {i}" for i in range(steps + 1)]
    return pd.DataFrame(stock, columns=cols), pd.DataFrame(option, columns=cols)


if __name__ == "__main__":
    # change these to whatever you want to test
    S, X, t, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20

    # 1) show a small tree so you can see the stock prices and the option values at every node
    steps_small = 4
    stock_tree, call_tree = crr_tree(S, X, t, r, q, sigma, steps_small, "call")
    u, d, p, disc = crr_parameters(t, r, q, sigma, steps_small)
    print(f"CRR parameters (N = {steps_small}): u = {u:.4f}, d = {d:.4f}, p = {p:.4f}")
    print("\nStock price tree (rows = number of up-moves):")
    print(stock_tree.round(2).to_string(na_rep=""))
    print("\nEuropean call value tree:")
    print(call_tree.round(2).to_string(na_rep=""))

    # 2) benchmark against Black-Scholes-Merton from bsm_formula.py
    d1, d2 = calc_d1_d2(S=S, X=X, t=t, r=r, q=q, sigma=sigma)
    bsm_c: float = bsm_call(S=S, X=X, t=t, r=r, q=q, d1=d1, d2=d2)
    bsm_p: float = bsm_put(S=S, X=X, t=t, r=r, q=q, d1=d1, d2=d2)
    print(f"\nBlack-Scholes-Merton call = {bsm_c:.4f}, put = {bsm_p:.4f}")

    # 3) convergence table: tree price vs BSM as N grows
    step_list = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    rows = []
    for n in step_list:
        tree_c = crr_price(S, X, t, r, q, sigma, n, "call")
        rows.append({"steps": n, "CRR call": tree_c, "BSM call": bsm_c, "abs error": abs(tree_c - bsm_c)})
    print("\nConvergence of the CRR call price to Black-Scholes-Merton:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4) put-call parity check using the tree prices (should match S*e^(-qt) - X*e^(-rt))
    n = 1000
    tree_call = crr_price(S, X, t, r, q, sigma, n, "call")
    tree_put = crr_price(S, X, t, r, q, sigma, n, "put")
    parity: float = put_call_parity(S=S, X=X, t=t, r=r, q=q)
    print(f"\nPut-Call parity: {parity:.4f} and the tree difference (C-P) {tree_call - tree_put:.4f} (N = {n})")

    # 5) the thing BSM can't do: American options (early exercise)
    amer_put = crr_price(S, X, t, r, q, sigma, n, "put", american=True)
    print(f"\nEuropean put = {tree_put:.4f}, American put = {amer_put:.4f} "
          f"(early exercise premium = {amer_put - tree_put:.4f})")

    # 6) optional plot of the convergence (saved next to this file)
    try:
        import matplotlib.pyplot as plt

        ns = np.arange(1, 201)
        prices = [crr_price(S, X, t, r, q, sigma, int(k), "call") for k in ns]

        plt.figure(figsize=(8, 4))
        plt.plot(ns, prices, label="CRR tree")
        plt.axhline(bsm_c, color="orange", linestyle="--", label="Black-Scholes-Merton")
        plt.xlabel("Number of steps N")
        plt.ylabel("European call price")
        plt.title("CRR binomial tree converging to Black-Scholes-Merton")
        plt.legend()
        plt.tight_layout()
        out = Path(__file__).resolve().parent / "crr_convergence.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved convergence plot to {out}")
        plt.show()
    except ImportError:
        print("\nmatplotlib not installed, skipping the plot (pip install matplotlib)")