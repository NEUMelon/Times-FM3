"""Unified base class for time series foundation models and baselines."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    @abstractmethod
    def forecast(
        self,
        ctx: np.ndarray,
        horizon: int,
        univariate: bool = False,
        use_symmetric_averaging: bool = True,
        return_quantiles: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Produce forecasts for a batch of series / channels.
        
        Args:
            ctx: Array of shape [num_variates, context_length]
            horizon: Prediction horizon length
            univariate: If True, treat each channel independently
            use_symmetric_averaging: If True, run forward on x and -x and average
            return_quantiles: If True, return 9 quantiles [N, H, 9]
            
        Returns:
            Tuple of (point_predictions [N, H], quantile_predictions [N, H, num_quantiles] or None)
        """
        pass
