"""Attention hooks and layer-wise knockout mechanisms for TimesFM-3."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class AttentionCollectorHook:
    """Hooks into variate attention layers to record attention weight matrices."""

    def __init__(self, target_layers: Optional[List[int]] = None):
        self.target_layers = target_layers
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.handles = []

    def register(self, model: nn.Module):
        # Locate MixingTransformer layers
        # In TimesFM-3: model.transformer.layers or similar
        layers = None
        if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            layers = model.transformer.layers
        elif hasattr(model, "model") and hasattr(model.model, "transformer"):
            layers = model.model.transformer.layers

        if layers is None:
            print("Warning: Could not locate transformer layers for attention hook.")
            return

        for idx, layer in enumerate(layers):
            if self.target_layers is None or idx in self.target_layers:
                # Find variate attention module in layer
                # TimesFM-3 layer typically has: time_attn, variate_attn, ffn
                if hasattr(layer, "variate_attn"):
                    h = layer.variate_attn.register_forward_hook(self._make_hook(idx))
                    self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(module, input_tensor, output_tensor):
            # Output of attention or hook into attention weights computation
            # If module outputs (context, weights)
            if isinstance(output_tensor, tuple) and len(output_tensor) > 1:
                self.attention_maps[layer_idx] = output_tensor[1].detach().cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.attention_maps = {}


class VariateAttentionKnockout:
    """
    Patches specified transformer layers to knockout cross-variate communication.
    When variate attention only attends to itself, it reduces to out_proj(value_proj(x)).
    """

    def __init__(self, model: nn.Module, knockout_layers: List[int]):
        self.model = model
        self.knockout_layers = set(knockout_layers)
        self.original_forwards = {}

    def apply(self):
        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            layers = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "transformer"):
            layers = self.model.model.transformer.layers

        if layers is None:
            return

        for idx, layer in enumerate(layers):
            if idx in self.knockout_layers and hasattr(layer, "variate_attn"):
                orig = layer.variate_attn.forward
                self.original_forwards[idx] = orig

                # Self-only bypass: x passes through value and out projection without cross-attention
                def make_bypass(mod):
                    def bypass_forward(x, *args, **kwargs):
                        # Value projection + output projection directly
                        if hasattr(mod, "v_proj") and hasattr(mod, "out_proj"):
                            v = mod.v_proj(x)
                            return mod.out_proj(v)
                        return x
                    return bypass_forward

                layer.variate_attn.forward = make_bypass(layer.variate_attn)

    def restore(self):
        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            layers = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "transformer"):
            layers = self.model.model.transformer.layers

        if layers is None:
            return

        for idx, orig in self.original_forwards.items():
            if idx < len(layers) and hasattr(layers[idx], "variate_attn"):
                layers[idx].variate_attn.forward = orig
        self.original_forwards = {}
