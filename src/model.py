"""Mixed-integer unit commitment and economic dispatch for a heating network.

For every hour t the model decides how much heat each unit produces and whether it is on:

    p_hp[t]   >= 0   electricity used by the heat pump      heat = COP[t] * p_hp[t]
    q_chp[t]  >= 0   heat from the CHP engine               electricity sold = r * q_chp[t]
    q_b[t]    >= 0   heat from the gas boiler
    ch[t], dis[t]    storage charge / discharge,  soc[t] storage level
    u_*[t] in {0,1}  unit on/off,   v_chp[t] >= 0  CHP start-up

min  sum_t  (spot[t] + fee) * p_hp[t]                              buy electricity
          + c_gas * (q_chp[t] / eta_chp + q_b[t] / eta_b)          buy gas incl. CO2 price
          - spot[t] * r * q_chp[t]                                 sell CHP electricity
          + c_start * v_chp[t]                                     start-up cost

s.t. COP[t] p_hp[t] + q_chp[t] + q_b[t] + dis[t] - ch[t] = demand[t]    heat balance
     soc[t] = (1 - loss) soc[t-1] + ch[t] - dis[t],   soc[t] <= cap      storage dynamics
     min_load * max * u[t] <= output[t] <= max * u[t]                  on/off with minimum load
     v_chp[t] >= u_chp[t] - u_chp[t-1]                                 start-up detection
     sum_{tau = t-L+1..t} v_chp[tau] <= u_chp[t]                       minimum up time of L hours
     soc[end] >= soc[start]                                            no "free" heat from the tank
"""
from dataclasses import dataclass

import numpy as np

from milp import MILPBuilder


@dataclass
class Dispatch:
    hp_heat: np.ndarray
    hp_el: np.ndarray
    chp_heat: np.ndarray
    boiler_heat: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    chp_on: np.ndarray
    starts: int


def solve_dispatch(plant, week, hours=None, storage_cap=None, soc0=None, chp_prev_on=0, chp_forced_on=False):
    """Solve the dispatch MILP over `hours` (default: the whole week)."""
    hours = np.arange(len(week.heat_demand)) if hours is None else np.asarray(hours)
    T = len(hours)
    cap = plant.storage_cap if storage_cap is None else storage_cap
    soc0 = 0.0 if soc0 is None else soc0  # start (and end at least) empty: no free heat from the tank
    c_gas = plant.gas_price + plant.co2_price * plant.gas_co2
    spot, cop, dem = week.spot_price[hours], week.cop[hours], week.heat_demand[hours]

    m = MILPBuilder()
    p_hp = np.array([m.add_var(cost=spot[t] + plant.grid_fee) for t in range(T)])
    q_chp = np.array([m.add_var(cost=c_gas / plant.chp_heat_eff - spot[t] * plant.chp_power_to_heat) for t in range(T)])
    q_b = m.add_vars(T, cost=c_gas / plant.boiler_eff)
    u_hp = m.add_vars(T, ub=1, integer=True)
    u_chp = m.add_vars(T, lb=1 if chp_forced_on else 0, ub=1, integer=True)
    v_chp = m.add_vars(T, cost=plant.chp_startup_cost, ub=1)
    u_b = m.add_vars(T, ub=1, integer=True)
    ch = m.add_vars(T, ub=min(plant.storage_rate, cap))
    dis = m.add_vars(T, ub=min(plant.storage_rate, cap))
    soc = m.add_vars(T, ub=cap)

    for t in range(T):
        m.add_constr({p_hp[t]: cop[t], q_chp[t]: 1, q_b[t]: 1, dis[t]: 1, ch[t]: -1}, lb=dem[t], ub=dem[t])
        # on/off with minimum load
        for q, u, qmax, mn in [(p_hp, u_hp, plant.hp_el_max, plant.hp_min_load),
                               (q_chp, u_chp, plant.chp_heat_max, plant.chp_min_load),
                               (q_b, u_b, plant.boiler_heat_max, plant.boiler_min_load)]:
            m.add_constr({q[t]: 1, u[t]: -qmax}, ub=0)
            m.add_constr({q[t]: 1, u[t]: -mn * qmax}, lb=0)
        # CHP start-up and minimum up time
        if t == 0:
            m.add_constr({v_chp[0]: 1, u_chp[0]: -1}, lb=-chp_prev_on)
        else:
            m.add_constr({v_chp[t]: 1, u_chp[t]: -1, u_chp[t - 1]: 1}, lb=0)
        window = {v_chp[k]: 1 for k in range(max(0, t - plant.chp_min_up + 1), t + 1)}
        window[u_chp[t]] = window.get(u_chp[t], 0) - 1
        m.add_constr(window, ub=0)
        # storage balance
        prev = {soc[t - 1]: -(1 - plant.storage_loss)} if t > 0 else {}
        rhs = (1 - plant.storage_loss) * soc0 if t == 0 else 0
        m.add_constr({soc[t]: 1, ch[t]: -1, dis[t]: 1, **prev}, lb=rhs, ub=rhs)
    m.add_constr({soc[T - 1]: 1}, lb=soc0)

    x = m.solve().x
    return Dispatch(hp_heat=cop * x[p_hp], hp_el=x[p_hp], chp_heat=x[q_chp], boiler_heat=x[q_b],
                    charge=x[ch], discharge=x[dis], soc=x[soc], chp_on=np.round(x[u_chp]),
                    starts=int(round(x[v_chp].sum())))


def solve_myopic(plant, week):
    """Rule-of-thumb controller: picks the cheapest units hour by hour, with no look-ahead and no storage."""
    parts, prev_on, run = [], 0, 0
    for t in range(len(week.heat_demand)):
        forced = prev_on == 1 and run < plant.chp_min_up  # respect minimum up time of a running CHP
        d = solve_dispatch(plant, week, hours=[t], storage_cap=0.0, soc0=0.0, chp_prev_on=prev_on, chp_forced_on=forced)
        on = int(d.chp_on[0])
        run = run + 1 if (on and prev_on) else (1 if on else 0)
        prev_on = on
        parts.append(d)
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])
    return Dispatch(*(cat(f) for f in ["hp_heat", "hp_el", "chp_heat", "boiler_heat", "charge", "discharge",
                                       "soc", "chp_on"]), starts=sum(p.starts for p in parts))


def evaluate(plant, week, d: Dispatch, future_grid_co2=0.15):
    """Cost breakdown [EUR] and emissions [t CO2] of a dispatch plan for the whole week.

    co2_tonnes uses today's grid mix; co2_tonnes_future_grid re-counts the same plan with a cleaner grid.
    """
    c_gas = plant.gas_price + plant.co2_price * plant.gas_co2
    gas = d.chp_heat / plant.chp_heat_eff + d.boiler_heat / plant.boiler_eff
    chp_el = plant.chp_power_to_heat * d.chp_heat
    out = {
        "electricity_purchase": float(np.sum((week.spot_price + plant.grid_fee) * d.hp_el)),
        "gas_and_co2": float(np.sum(c_gas * gas)),
        "chp_electricity_revenue": float(-np.sum(week.spot_price * chp_el)),
        "chp_startups": float(plant.chp_startup_cost * d.starts),
    }
    out["total_cost"] = sum(out.values())
    # Net CO2: gas burned + grid electricity bought - grid electricity displaced by the CHP (standard credit method)
    net_grid = np.sum(d.hp_el) - np.sum(chp_el)
    out["co2_tonnes"] = float(np.sum(gas) * plant.gas_co2 + net_grid * plant.grid_co2)
    out["co2_tonnes_future_grid"] = float(np.sum(gas) * plant.gas_co2 + net_grid * future_grid_co2)
    total_heat = float(np.sum(week.heat_demand))
    for name, q in [("share_heat_pump", d.hp_heat), ("share_chp", d.chp_heat), ("share_boiler", d.boiler_heat)]:
        out[name] = max(0.0, float(np.sum(q))) / total_heat
    out["chp_starts"] = d.starts
    out["cost_per_mwh_heat"] = out["total_cost"] / total_heat
    return out
