from collections import namedtuple
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.typing import ColorType

from volatility_drag.gbm import gbm_with_annual_params
from volatility_drag.leverage import leverage_asset

n_steps = 365 * 5
initial_value = 100
number_paths = 100
alpha = 0.3
linewidth = 0.35
linewidth_averaged_path = 4

Scenario = namedtuple("Scenario", ["volatility", "expected_return"])
scenarios = [
    # Scenario(volatility=0.08, expected_return=0.06), # Low volatility, low return
    Scenario(volatility=0.15, expected_return=0.08),  # Roughly S&P 500
    # Scenario(volatility=0.20, expected_return=0.04), # High volatility, low return
    # Scenario(volatility=0.20, expected_return=0.12),  # High volatility, high return
]

leverage_factors: list[float] = [0.5, 1.0, 2.0]


def generate_path(volatility: float, expected_return: float, leverage: float) -> list[float]:
    """Generates a leveraged GBM path."""
    gbm_path = gbm_with_annual_params(n_steps, initial_value, volatility, expected_return)
    return leverage_asset(gbm_path, leverage)


def main() -> None:
    plt.figure(figsize=(12, 8))
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_steps + 1)]

    label_colors: dict[str, ColorType] = {}
    for scenario in scenarios:
        for lever in leverage_factors:
            label = (
                f"Volatility: {scenario.volatility:.0%}, "
                f"Expected Return: {scenario.expected_return:.0%}"
            )
            if lever != 1.0:
                label += f", Leverage: {lever:.1f}x"

            paths = []
            for _ in range(number_paths):
                path = generate_path(scenario.volatility, scenario.expected_return, lever)
                paths.append(path)
                # Plot individual paths with alpha
                if label not in label_colors:
                    (line,) = plt.plot(dates, path, label=label, alpha=alpha, linewidth=linewidth)
                    label_colors[label] = line.get_color()
                else:
                    plt.plot(
                        dates, path, color=label_colors[label], alpha=alpha, linewidth=linewidth
                    )
            median_path = [np.median(p) for p in zip(*paths, strict=False)]
            plt.plot(
                dates,
                median_path,
                color=label_colors[label],
                linewidth=linewidth_averaged_path,
                alpha=0.9,
            )

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.ylim(bottom=0, top=min(plt.ylim()[1], 2.0 * initial_value))
    plt.title("Simulated Asset Prices (Median Paths in Bold)")
    plt.xlabel("Date")
    plt.ylabel("Asset Price")
    legend = plt.legend()
    for line in legend.get_lines():
        line.set_linewidth(1)
        line.set_alpha(1)
    plt.gcf().autofmt_xdate()
    plt.grid(True, which="both")
    plt.show()


if __name__ == "__main__":
    main()
