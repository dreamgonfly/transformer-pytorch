from __future__ import annotations
from enum import Enum
from typing import List

import torch
from torch import Tensor


class AttentionMode(Enum):
    SELF = "self"
    MEMORY = "memory"


class AttentionCache(dict):
    @property
    def key_projected(self) -> Tensor:
        return self.get("key_projected")

    @key_projected.setter
    def key_projected(self, key_projected: Tensor) -> None:
        self["key_projected"] = key_projected

    @property
    def value_projected(self) -> Tensor:
        return self.get("value_projected")

    @value_projected.setter
    def value_projected(self, value_projected: Tensor) -> None:
        self["value_projected"] = value_projected

    def select_sample(self, sample_index: int) -> AttentionCache:
        cache = AttentionCache()
        if self.key_projected is not None:
            cache.key_projected = self.key_projected[sample_index : sample_index + 1]
        if self.value_projected is not None:
            cache.value_projected = self.value_projected[sample_index : sample_index + 1]
        return cache

    @classmethod
    def merge(cls, caches: List[AttentionCache]) -> AttentionCache:
        cache = AttentionCache()
        if caches[0].key_projected is not None:
            cache.key_projected = torch.cat([cache.key_projected for cache in caches])
        if caches[0].value_projected is not None:
            cache.value_projected = torch.cat([cache.value_projected for cache in caches])
        return cache


class AttentionState(dict):
    @property
    def attention(self) -> Tensor:
        return self.get("attention")

    @attention.setter
    def attention(self, attention: Tensor) -> None:
        self["attention"] = attention

    @property
    def cache(self) -> AttentionCache:
        if "cache" not in self:
            self["cache"] = AttentionCache()
        return self["cache"]

    @cache.setter
    def cache(self, new_cache: AttentionCache):
        self["cache"] = new_cache

    def select_sample(self, sample_index: int) -> AttentionState:
        state = AttentionState()
        if self.attention is not None:
            state.attention = self.attention[sample_index : sample_index + 1]
        state.cache = self.cache.select_sample(sample_index)
        return state

    @classmethod
    def merge(cls, states: List[AttentionState]) -> AttentionState:
        state = AttentionState()
        if states[0].attention is not None:
            state.attention = torch.cat([state.attention for state in states])
        if states[0].cache is not None:
            state.cache = AttentionCache.merge([state.cache for state in states])
        return state


class LayerState(dict):
    @property
    def self_attention(self) -> AttentionState:
        if "self-attention" not in self:
            self["self-attention"] = AttentionState()
        return self["self-attention"]

    @self_attention.setter
    def self_attention(self, self_attention: AttentionState) -> None:
        self["self-attention"] = self_attention

    @property
    def memory_attention(self) -> AttentionState:
        if "memory-attention" not in self:
            self["memory-attention"] = AttentionState()
        return self["memory-attention"]

    @memory_attention.setter
    def memory_attention(self, memory_attention: Tensor) -> None:
        self["memory-attention"] = memory_attention

    def select_sample(self, sample_index: int) -> LayerState:
        state = LayerState()
        state.self_attention = self.self_attention.select_sample(sample_index)
        state.memory_attention = self.memory_attention.select_sample(sample_index)
        return state

    @classmethod
    def merge(cls, states: List[LayerState]):
        state = LayerState()
        state.self_attention = AttentionState.merge([state.self_attention for state in states])
        state.memory_attention = AttentionState.merge([state.memory_attention for state in states])
        return state


class EncoderState(dict):
    def select_layer(self, layer_index: int) -> LayerState:
        if layer_index not in self:
            self[layer_index] = LayerState()
        return self[layer_index]

    def set_layer(self, layer_index, layer_state: LayerState) -> None:
        self[layer_index] = layer_state


class DecoderState(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["layers"] = {}
        self["position"] = 0

    @property
    def position(self):
        return self["position"]

    @property
    def layers(self):
        return self["layers"]

    @position.setter
    def position(self, value: int):
        self["position"] = value

    def select_layer(self, layer_index: int) -> LayerState:
        if layer_index not in self.layers:
            self.layers[layer_index] = LayerState()
        return self.layers[layer_index]

    def set_layer(self, layer_index, layer_state: LayerState) -> None:
        self.layers[layer_index] = layer_state

    def select_sample(self, sample_index: int) -> DecoderState:
        state = DecoderState()
        state.position = self.position
        for layer_index, layer_state in self.layers.items():
            state.set_layer(layer_index, layer_state.select_sample(sample_index))
        return state

    @classmethod
    def merge(cls, states: List[DecoderState]):
        state = DecoderState()
        state.position = states[0].position
        for layer_index in states[0].layers.keys():
            state.set_layer(
                layer_index, LayerState.merge([state.select_layer(layer_index) for state in states])
            )
        return state
