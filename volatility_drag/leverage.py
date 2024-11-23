import numpy as np
import numpy.typing as npt


def leverage_asset(path: npt.NDArray[np.floating], lever: float) -> npt.NDArray[np.floating]:
    """
    Applies leverage to a simulated asset path.

    Args:
        path: A 1D array of asset prices.
        lever: The leverage factor.

    Returns:
        A 1D array of leveraged asset prices.
    """
    pct_change = np.diff(path, axis=0) / path[:-1]
    leveraged_pct_change = pct_change * lever
    leveraged_path = np.cumprod(1 + leveraged_pct_change) * path[0]
    leveraged_path = np.insert(leveraged_path, 0, path[0])
    return leveraged_path
