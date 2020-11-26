from __future__ import annotations
from enum import Enum
from typing import List

from torch import Tensor


class AttentionMode(Enum):
    SELF = "self"
    MEMORY = "memory"


class AttentionCache(dict):
    @property
    def key_projected(self) -> Tensor:
        return self["key_projected"]

    @key_projected.setter
    def key_projected(self, key_projected: Tensor) -> None:
        self["key_projected"] = key_projected

    @property
    def value_projected(self) -> Tensor:
        return self["value_projected"]

    @value_projected.setter
    def value_projected(self, value_projected: Tensor) -> None:
        self["value_projected"] = value_projected


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


class EncoderState(dict):
    def select_layer(self, layer_index: int) -> LayerState:
        if layer_index not in self:
            self[layer_index] = LayerState()
        return self[layer_index]

    def set_layer(self, layer_index, layer_state: LayerState) -> None:
        self[layer_index] = layer_state


class DecoderState(dict):
    def select_layer(self, layer_index: int) -> LayerState:
        if layer_index not in self:
            self[layer_index] = LayerState()
        return self[layer_index]

    def set_layer(self, layer_index, layer_state: LayerState) -> None:
        self[layer_index] = layer_state

    def select_sample(self, sample_index: int):
        pass

    @classmethod
    def merge(cls, states: List[DecoderState]):
        pass


# class LayerCache(dict):
#     @property
#     def self(self) -> AttentionCache:
#         return self[CacheMode.SELF.value]
#
#     @property
#     def memory(self) -> AttentionCache:
#         return self[CacheMode.MEMORY.value]
#
#     @classmethod
#     def new(cls, self_attention_cache: AttentionCache, memory_attention_cache: AttentionCache):
#         return cls(self=self_attention_cache, memory=memory_attention_cache)
#
#
# class Cache(dict):
#     def select_layer(self, layer_index: int) -> LayerCache:
#         return self.get(layer_index)
#
#     def set_layer(self, layer_index: int, layer_cache: LayerCache) -> None:
#         self[layer_index] = layer_cache
#
#
# class LayerAttention(dict):
#     @classmethod
#     def new(cls, self_attention: Tensor, memory_attention: Tensor):
#         return cls(self=self_attention, memory=memory_attention)
