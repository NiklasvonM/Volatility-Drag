from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from volatility_drag.gbm import gbm_with_annual_params
from volatility_drag.leverage import leverage_asset

n_years = 5
initial_value = 100
number_paths = 10000
leverage_factor = 2.0

Scenario = namedtuple("Scenario", ["volatility", "expected_return"])

volatilities = np.arange(0.05, 0.31, 0.01)
expected_returns = np.arange(0.00, 0.16, 0.01)

scenarios = [
    Scenario(volatility=vol, expected_return=ret)
    for vol in volatilities
    for ret in expected_returns
]


def generate_path(
    volatility: float, expected_return: float, leverage: float
) -> npt.NDArray[np.floating]:
    """Generates a leveraged GBM path."""
    gbm_path = gbm_with_annual_params(n_years * 365, initial_value, volatility, expected_return)
    return leverage_asset(gbm_path, leverage)


def main() -> None:
    results = np.zeros((len(volatilities), len(expected_returns)))

    for i, vol in enumerate(tqdm(volatilities)):
        for j, ret in enumerate(expected_returns):
            paths_leveraged = []
            paths_unleveraged = []
            for _ in range(number_paths):
                path_leveraged = generate_path(vol, ret, leverage_factor)
                paths_leveraged.append(path_leveraged[-1])
                path_unleveraged = generate_path(vol, ret, 1.0)
                paths_unleveraged.append(path_unleveraged[-1])
            median_leveraged = np.median(paths_leveraged)
            median_unleveraged = np.median(paths_unleveraged)
            results[i, j] = median_leveraged / median_unleveraged

    _, ax = plt.subplots(figsize=(10, 8))
    my_colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red, white, blue
    cmap = LinearSegmentedColormap.from_list("red_white_blue", my_colors)
    divnorm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    im = ax.pcolormesh(
        expected_returns,
        volatilities,
        results,
        cmap=cmap,
        norm=divnorm,
        shading="auto",
    )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        f"Median Final Value Ratio (Leverage {leverage_factor:.1f}x / Unleveraged)",
        rotation=-90,
        va="bottom",
    )

    ax.set_xticks(expected_returns)
    ax.set_yticks(volatilities)
    ax.set_xlabel("Expected Annual Return")
    ax.set_ylabel("Annual Volatility")
    plt.title(
        f"Effect of Leverage ({leverage_factor:.1f}x) on Median Final Value\n"
        f"({n_years} years, {number_paths} simulations each)"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
