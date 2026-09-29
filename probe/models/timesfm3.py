"""TimesFM 3.0 wrapper supporting both Univariate and Multivariate modes with official chunking."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch import timesfm3_forecaster as tf_module
from timesfm3.torch.evaluator import TimesFM3Evaluator, _MAX_VARIATES_PER_FORWARD
from probe.config import MODEL_DIR, QUANTILES
from probe.models.base import BaseForecaster


class TimesFM3Wrapper(BaseForecaster):
    """
    High-level wrapper around TimesFM 3.0 PyTorch implementation.
    Enforces make_positive=False for normalized time series,
    and supports explicit univariate vs multivariate toggling,
    multi-window batching, and monkeypatches try_gc to eliminate CPU pauses.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        self.checkpoint_path = str(checkpoint_path or MODEL_DIR)
        self.device = device
        
        # Monkeypatch try_gc in timesfm3 to no-op
        tf_module.try_gc = lambda *args, **kwargs: None
        
        print(f"Loading TimesFM-3.0 Evaluator from {self.checkpoint_path} on {device}...")
        self.evaluator = TimesFM3Evaluator(
            checkpoint_path=self.checkpoint_path,
            device=device,
        )
        self.forecaster = self.evaluator
        print("TimesFM-3.0 successfully initialized (try_gc monkeypatched).")

    def forecast(
        self,
        ctx: Union[np.ndarray, List[np.ndarray]],
        horizon: int,
        univariate: bool = False,
        use_symmetric_averaging: bool = True,
        return_quantiles: bool = True,
        bypass_evaluator_chunking: bool = False,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], Tuple[List[np.ndarray], Optional[List[np.ndarray]]]]:
        """
        Produce forecasts for a single window [num_variates, context_length]
        or a batch of windows List[[num_variates, context_length]].
        If bypass_evaluator_chunking is True, bypasses the official 32-channel
        chunking in TimesFM3Evaluator and directly invokes TimesFM3Forecaster to
        perform a single, unchunked forward pass with all variates attending jointly.
        """
        is_single = isinstance(ctx, np.ndarray) and ctx.ndim == 2
        contexts_list = [ctx] if is_single else list(ctx)

        # Dynamic per_core_batch_size tuning
        if univariate:
            self.evaluator.config.per_core_batch_size = 512
        else:
            self.evaluator.config.per_core_batch_size = 32

        if bypass_evaluator_chunking:
            if univariate:
                raise ValueError("bypass_evaluator_chunking is strictly for multivariate (K > 32); univariate mode must not bypass evaluator.")

            # Ensure forecaster per_core_batch_size is configured identical to evaluator
            self.evaluator.config.per_core_batch_size = 32

            # Bypass official 32-channel chunking in TimesFM3Evaluator by calling TimesFM3Forecaster directly
            if issubclass(self.evaluator.__class__, TimesFM3Forecaster):
                outs = list(TimesFM3Forecaster.predict_batch(
                    self.evaluator,
                    contexts=contexts_list,
                    horizon=horizon,
                    return_quantiles=return_quantiles,
                    use_symmetric_averaging=use_symmetric_averaging,
                    make_positive=False,
                ))
            else:
                if not hasattr(self, "_raw_forecaster") or self._raw_forecaster is None:
                    self._raw_forecaster = TimesFM3Forecaster(
                        checkpoint_path=self.checkpoint_path,
                        device=self.device,
                    )
                self._raw_forecaster.config.per_core_batch_size = 32
                outs = list(self._raw_forecaster.predict_batch(
                    contexts=contexts_list,
                    horizon=horizon,
                    return_quantiles=return_quantiles,
                    use_symmetric_averaging=use_symmetric_averaging,
                    make_positive=False,
                ))
        else:
            outs = list(self.evaluator.predict_batch(
                contexts=contexts_list,
                horizon=horizon,
                univariate=univariate,
                return_quantiles=return_quantiles,
                use_symmetric_averaging=use_symmetric_averaging,
                make_positive=False,  # CRITICAL: normalized LTSF data has negative values
            ))

        point_preds = [o.forecast for o in outs]
        quant_preds = [o.quantiles for o in outs] if return_quantiles else None

        if is_single:
            return point_preds[0], (quant_preds[0] if quant_preds is not None else None)
        return point_preds, quant_preds
