"""
Image feature cache for MatAnyone2 inference.
"""

import warnings

import torch

from ..model.matanyone2 import MatAnyone2


class ImageFeatureStore:
    """Caches image features for reuse during inference."""

    def __init__(self, network: MatAnyone2, no_warning: bool = False):
        self.network = network
        self._store = {}
        self.no_warning = no_warning

    def _encode_feature(self, index: int, image: torch.Tensor, last_feats=None) -> None:
        ms_features, pix_feat = self.network.encode_image(image, last_feats=last_feats)
        key, shrinkage, selection = self.network.transform_key(ms_features[0])
        self._store[index] = (ms_features, pix_feat, key, shrinkage, selection)

    def get_all_features(self, images: torch.Tensor) -> tuple:
        seq_length = images.shape[0]
        ms_features, pix_feat = self.network.encode_image(images, seq_length)
        key, shrinkage, selection = self.network.transform_key(ms_features[0])
        for index in range(seq_length):
            self._store[index] = (
                [f[index].unsqueeze(0) for f in ms_features],
                pix_feat[index].unsqueeze(0),
                key[index].unsqueeze(0),
                shrinkage[index].unsqueeze(0),
                selection[index].unsqueeze(0),
            )

    def get_features(self, index: int, image: torch.Tensor, last_feats=None) -> tuple:
        if index not in self._store:
            self._encode_feature(index, image, last_feats)
        return self._store[index][:2]

    def get_key(self, index: int, image: torch.Tensor, last_feats=None) -> tuple:
        if index not in self._store:
            self._encode_feature(index, image, last_feats)
        return self._store[index][2:]

    def delete(self, index: int) -> None:
        if index in self._store:
            del self._store[index]

    def __len__(self):
        return len(self._store)

    def __del__(self):
        if len(self._store) > 0 and not self.no_warning:
            warnings.warn(
                "Leaking %s in the image feature store" % list(self._store.keys())
            )
