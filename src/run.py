"""Experiments: optimal vs. rule-of-thumb dispatch, value of thermal storage, and CO2 price sensitivity.

Usage:  python src/run.py     (from the project root). Outputs go to results/.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import Plant, make_week
from model import evaluate, solve_dispatch, solve_myopic

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
COLORS = {"hp": "#2e86ab", "chp": "#f18f01", "boiler": "#c73e1d", "storage": "#6a994e"}


def main():
    plant = Plant()
    week = make_week(plant)

    # 1) Three ways to run the same plant
    print("Solving optimal dispatch, optimal without storage, and hour-by-hour controller ...")
    opt = solve_dispatch(plant, week)
    nostore = solve_dispatch(plant, week, storage_cap=0.0)
    myopic = solve_myopic(plant, week)
    strategies = {"Optimal (MILP, 1-week look-ahead)": opt,
                  "Optimal without storage": nostore,
                  "Hour-by-hour controller": myopic}
    comp = pd.DataFrame({k: evaluate(plant, week, d) for k, d in strategies.items()}).T
    comp.to_csv(OUT / "strategy_comparison.csv")

    # 2) Value of storage: how much does each MWh of tank save?
    print("Storage size sweep ...")
    rows = []
    for cap in [0, 1, 2, 4, 6, 8, 12]:
        d = solve_dispatch(plant, week, storage_cap=float(cap))
        rows.append({"storage_mwh": cap, **evaluate(plant, week, d)})
    storage = pd.DataFrame(rows)
    storage["saving_vs_no_storage"] = storage["total_cost"].iloc[0] - storage["total_cost"]
    storage.to_csv(OUT / "storage_sweep.csv", index=False)

    # 3) CO2 price sensitivity: when does the plant switch from gas to the heat pump?
    print("CO2 price sweep ...")
    rows = []
    for co2 in [0, 55, 100, 150, 200, 300, 400]:
        p = replace(plant, co2_price=float(co2))
        rows.append({"co2_price": co2, **evaluate(p, week, solve_dispatch(p, week))})
    co2 = pd.DataFrame(rows)
    co2.to_csv(OUT / "co2_price_sweep.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    fmt = lambda v: f"{v:,.2f}"
    print("\n=== Strategy comparison (one winter week) ===")
    print(comp[["total_cost", "cost_per_mwh_heat", "co2_tonnes", "co2_tonnes_future_grid", "share_heat_pump", "share_chp", "share_boiler",
                "chp_starts"]].to_string(float_format=fmt))
    print("\n=== Storage size sweep ===")
    print(storage[["storage_mwh", "total_cost", "saving_vs_no_storage", "chp_starts"]].to_string(index=False, float_format=fmt))
    print("\n=== CO2 price sweep ===")
    print(co2[["co2_price", "total_cost", "co2_tonnes", "co2_tonnes_future_grid", "share_heat_pump", "share_chp",
               "share_boiler"]]
          .to_string(index=False, float_format=fmt))

    plot_dispatch(week, opt, "Optimal dispatch (MILP with storage)", OUT / "dispatch_optimal.png")
    plot_dispatch(week, myopic, "Hour-by-hour controller (no look-ahead, no storage)", OUT / "dispatch_myopic.png")
    plot_sweeps(storage, co2)
    print(f"\nResults written to {OUT}")


def plot_dispatch(week, d, title, path):
    h = np.arange(len(week.heat_demand))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    stack = [d.hp_heat, d.chp_heat, d.boiler_heat, d.discharge]
    labels = ["Heat pump", "CHP", "Gas boiler", "Storage discharge"]
    cols = [COLORS["hp"], COLORS["chp"], COLORS["boiler"], COLORS["storage"]]
    a1.stackplot(h, *stack, labels=labels, colors=cols, step="post", alpha=0.9)
    a1.step(h, week.heat_demand + d.charge, where="post", color="black", lw=1, label="Demand + storage charging")
    a1.set_ylabel("Heat [MW]")
    a1.set_title(title)
    a1.legend(loc="upper left", fontsize=8, ncol=5)
    a1.set_ylim(0, max(3.0, (week.heat_demand + d.charge).max() * 1.15))
    b = a1.twinx()
    b.plot(h, week.spot_price, color="#555555", lw=0.8, ls="--")
    b.set_ylabel("Electricity price [EUR/MWh]", color="#555555")
    a2.fill_between(h, d.soc, step="post", color=COLORS["storage"], alpha=0.6)
    a2.set_ylabel("Storage [MWh]")
    a2.set_xlabel("Hour of the week")
    a2.set_xticks(range(0, 169, 24))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweeps(storage, co2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(storage["storage_mwh"], storage["saving_vs_no_storage"], "o-", color=COLORS["storage"])
    a1.set_xlabel("Storage size [MWh]")
    a1.set_ylabel("Weekly saving vs. no storage [EUR]")
    a1.set_title("Value of thermal storage (diminishing returns)")
    a1.grid(alpha=0.3)
    x = np.arange(len(co2))
    a2.bar(x, co2["share_heat_pump"] * 100, color=COLORS["hp"], label="Heat pump")
    a2.bar(x, co2["share_chp"] * 100, bottom=co2["share_heat_pump"] * 100, color=COLORS["chp"], label="CHP")
    a2.bar(x, co2["share_boiler"] * 100, bottom=(co2["share_heat_pump"] + co2["share_chp"]) * 100,
           color=COLORS["boiler"], label="Gas boiler")
    a2.set_xticks(x, co2["co2_price"].astype(int))
    a2.set_xlabel("CO2 price [EUR/t]")
    a2.set_ylabel("Share of heat [%]")
    a2.set_title("Heat mix vs. CO2 price (net CO2 today / future grid, t)")
    for i, (t, f) in enumerate(zip(co2["co2_tonnes"], co2["co2_tonnes_future_grid"])):
        a2.text(i, 103, f"{t:.0f} / {f:.0f}", ha="center", fontsize=7.5)
    a2.set_ylim(0, 112)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
