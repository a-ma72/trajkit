"""trajkit — GPS track conditioning for high-rate vehicle measurement data.

Sensor fusion (EKF/RTS), clothoid curvature analysis, and GPS signal
conditioning in a single, composable pipeline.
"""

from .clothoid import (
    G2ClothoidApproximator,
    G2ClothoidConfig,
    G2ClothoidFitResult,
)
from .kalman import (
    GPSKalmanFilter,
    KalmanConfig,
    KalmanResult,
)
from .processor import (
    EARTH_METERS_PER_DEGREE,
    GPSProcessor,
    GPSTrack,
)

__all__ = [
    "EARTH_METERS_PER_DEGREE",
    "G2ClothoidApproximator",
    "G2ClothoidConfig",
    "G2ClothoidFitResult",
    "GPSKalmanFilter",
    "GPSProcessor",
    "GPSTrack",
    "KalmanConfig",
    "KalmanResult",
]
