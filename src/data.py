"""Synthetic but realistic data for one winter week in a small decentralized heating network.

All values are hourly. Heat in MW_th, electricity in MW_el, energy in MWh, prices in EUR/MWh.
"""
from dataclasses import dataclass

import numpy as np

HOURS = 168  # one week


@dataclass
class Plant:
    # Heat pump (air-source): electricity in, heat out with temperature-dependent COP
    hp_el_max: float = 0.45        # MW_el
    hp_min_load: float = 0.25      # share of max when running
    hp_sink_temp: float = 60.0     # network supply temperature [deg C]
    hp_carnot_eff: float = 0.45    # share of the ideal Carnot COP that a real heat pump reaches
    grid_fee: float = 60.0         # EUR/MWh grid charges and levies on purchased electricity
    # Combined heat and power (gas engine): sells electricity at the spot price
    chp_heat_max: float = 1.0      # MW_th
    chp_min_load: float = 0.5
    chp_power_to_heat: float = 0.85
    chp_heat_eff: float = 0.45     # MW_th per MW of gas
    chp_startup_cost: float = 120.0
    chp_min_up: int = 4            # hours it must stay on after a start
    # Gas boiler: backup and peak load
    boiler_heat_max: float = 1.6
    boiler_min_load: float = 0.2
    boiler_eff: float = 0.92
    # Thermal storage tank
    storage_cap: float = 4.0       # MWh_th
    storage_rate: float = 1.0      # MW_th charge / discharge limit
    storage_loss: float = 0.005    # share of stored heat lost per hour
    # Fuel and emissions
    gas_price: float = 40.0        # EUR/MWh gas
    gas_co2: float = 0.201         # t CO2 per MWh gas
    grid_co2: float = 0.38         # t CO2 per MWh grid electricity (German average mix)
    co2_price: float = 55.0        # EUR/t CO2 (German national CO2 price for heating)


@dataclass
class Week:
    outdoor_temp: np.ndarray   # deg C
    heat_demand: np.ndarray    # MW_th
    spot_price: np.ndarray     # EUR/MWh_el
    cop: np.ndarray            # heat pump coefficient of performance


def make_week(plant: Plant, seed: int = 7) -> Week:
    rng = np.random.default_rng(seed)
    t = np.arange(HOURS)
    hod = t % 24

    # Outdoor temperature: daily cycle plus a cold front mid-week
    front = -5.0 * np.exp(-((t - 80) / 30.0) ** 2)
    temp = 1.0 + 4.0 * np.sin((hod - 9) / 24 * 2 * np.pi) + front + rng.normal(0, 0.8, HOURS)

    # Heat demand: space heating grows as it gets colder, plus hot water and morning/evening peaks
    peaks = 1 + 0.18 * np.exp(-((hod - 7) / 1.5) ** 2) + 0.12 * np.exp(-((hod - 19) / 2.0) ** 2)
    night = np.where((hod >= 23) | (hod < 5), 0.85, 1.0)
    demand = (0.30 + 0.065 * np.clip(16 - temp, 0, None)) * peaks * night
    demand = np.round(demand, 3)

    # Day-ahead electricity price: morning and evening peaks, solar dip at noon, wind noise
    sunny = rng.random(7) < 0.5
    solar_dip = np.array([40.0 if sunny[d] else 10.0 for d in t // 24]) * np.exp(-((hod - 13) / 2.5) ** 2)
    wind = np.zeros(HOURS)
    for i in range(1, HOURS):
        wind[i] = 0.92 * wind[i - 1] + rng.normal(0, 7)
    price = (85 + 45 * np.exp(-((hod - 8) / 1.8) ** 2) + 55 * np.exp(-((hod - 18.5) / 2.0) ** 2)
             - 25 * ((hod >= 0) & (hod < 5)) - solar_dip + wind)
    price = np.round(np.clip(price, -10, None), 1)

    # Heat pump COP from a Carnot model: COP = eff * T_sink / (T_sink - T_source) in Kelvin
    cop = plant.hp_carnot_eff * (plant.hp_sink_temp + 273.15) / (plant.hp_sink_temp - temp)
    return Week(temp, demand, price, cop)
