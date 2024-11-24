"""
Show path dependence by plotting scenarios with different volatilities but
the same, deterministic endpoint.
Even though the different assets reach the same price at the same time,
the leveraged versions vary.
"""

from collections import namedtuple
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from volatility_drag.gbm import brownian_bridge_with_drift
from volatility_drag.leverage import leverage_asset

n_steps = 365 * 5
initial_value = 100
alpha = 1.0
linewidth = 1.0

Scenario = namedtuple("Scenario", ["volatility", "expected_return"])
scenarios = [
    Scenario(volatility=0.05, expected_return=0.08),
    Scenario(volatility=0.15, expected_return=0.08),
    Scenario(volatility=0.30, expected_return=0.08),
]


def generate_path(volatility: float, expected_return: float, leverage: float) -> list[float]:
    """Generates a leveraged GBM path."""
    path = brownian_bridge_with_drift(n_steps, initial_value, volatility, expected_return)
    return leverage_asset(path, leverage)


def main() -> None:
    axes: Axes
    _, axes = plt.subplots(len(scenarios), 1, figsize=(10, 12))
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_steps + 1)]

    for i, scenario in enumerate(scenarios):
        ax = axes[i] if len(scenarios) > 1 else axes
        path = brownian_bridge_with_drift(
            n_steps, initial_value, scenario.volatility, scenario.expected_return
        )
        ax.plot(dates, path, label="Original Path", alpha=alpha, linewidth=linewidth)
        leveraged_path = leverage_asset(path, lever=2.0)
        ax.plot(
            dates,
            leveraged_path,
            label="2x Leveraged",
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_ylim(bottom=0, top=3 * initial_value)
        ax.set_title(f"Simulated Asset Prices (Path {i + 1})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Asset Price")
        legend = ax.legend()
        for line in legend.get_lines():
            line.set_linewidth(1)
            line.set_alpha(1)
        plt.gcf().autofmt_xdate()
        ax.set(xticklabels=[])
        ax.grid(True, which="both")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
