import numpy as np
import numpy.typing as npt


def geometric_brownian_motion(
    n_steps: int, dt: float, initial_value: float, drift: float, volatility: float
) -> npt.NDArray[np.floating]:
    """
    Simulates a geometric Brownian motion path.
    Source:
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Simulating_sample_paths

    Args:
        n_steps: The number of time steps to simulate.
        dt: The time step size.
        initial_value: The initial value of the Brownian motion.
        drift: The drift rate of the Brownian motion.
        volatility: The volatility of the Brownian motion.

    Returns:
        A 1D array of Brownian motion values.
    """

    x = np.exp(
        (drift - volatility**2 / 2) * dt
        + volatility * np.random.normal(0, np.sqrt(dt), size=(1, n_steps)).T
    )
    x = np.vstack([np.ones(1), x])
    x = initial_value * x.cumprod(axis=0)
    return x


def gbm_with_annual_params(
    n_steps: int, initial_value: float, volatility: float, expected_annual_return: float
) -> npt.NDArray[np.floating]:
    """
    Simulates a Geometric Brownian Motion path using annual parameters.

    Args:
        n_steps: The number of time steps (days) to simulate.
        initial_value: The initial value of the asset.
        annual_volatility: The annual volatility of the asset.
        expected_annual_return: The expected annual return of the asset.

    Returns:
        A 1D array of asset prices following a GBM.
    """

    dt = 1 / 365  # Daily time step
    drift = np.log(expected_annual_return + 1)
    return geometric_brownian_motion(n_steps, dt, initial_value, drift, volatility)
