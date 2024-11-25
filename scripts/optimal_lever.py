from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from volatility_drag.gbm import gbm_with_annual_params
from volatility_drag.leverage import leverage_asset

n_years = 5
number_paths = 10000
levers = np.arange(start=0.00, stop=3.5, step=0.01)
Scenario = namedtuple("Scenario", ["volatility", "expected_return"])
scenarios = [
    Scenario(volatility=0.15, expected_return=0.08),
    Scenario(volatility=0.15, expected_return=0.08 - 0.0325),  # S&P 500 minus ECB's interest rate
    Scenario(volatility=0.20, expected_return=0.08),
    Scenario(volatility=0.20, expected_return=0.05),
    Scenario(volatility=0.30, expected_return=0.03),
    Scenario(volatility=0.30, expected_return=0.10),
]


def main() -> None:
    for scenario in tqdm(scenarios):
        # Maps levers to the final price after n_years.
        final_values: defaultdict[float, list[float]] = defaultdict(list)
        for _ in range(number_paths):
            gbm_path = gbm_with_annual_params(
                n_years * 365, 100.0, scenario.volatility, scenario.expected_return
            )
            for lever in levers:
                leveraged_path = leverage_asset(gbm_path, lever)
                final_values[lever].append(leveraged_path[-1])

        medians = {lever: np.median(final_values) for lever, final_values in final_values.items()}
        label = (
            f"Volatility: {scenario.volatility:.0%}, "
            f"Expected Return: {scenario.expected_return:.2%}"
        )
        plt.plot(medians.keys(), medians.values(), label=label)
        best_lever = max(medians, key=medians.get)
        max_median = medians[best_lever]
        plt.plot(best_lever, max_median, "o", color="black")
    plt.title(
        f"Median Final Value After {n_years} Years for Different Levers and Selected Scenarios"
    )
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
