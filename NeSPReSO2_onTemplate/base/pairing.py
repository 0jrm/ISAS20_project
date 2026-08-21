"""Fail closed when a run mixes the wrong cache, checkpoint, or dataset tag."""

from __future__ import annotations

from typing import Any, Mapping


def assert_dataset_tags(config_tag: str | None, cache_tag: str | None) -> None:
    """Config io.dataset_tag must match the pickle it is about to train/eval on."""
    if not config_tag or not cache_tag:
        return
    if config_tag != cache_tag:
        raise ValueError(
            f"dataset_tag mismatch: config={config_tag!r} cache={cache_tag!r}. "
            "Pair the checkpoint with the cache it was trained on."
        )


def _pca_n_components(pca: Any) -> int | None:
    n = getattr(pca, "n_components_", None)
    if n is not None:
        return int(n)
    n = getattr(pca, "n_components", None)
    return int(n) if isinstance(n, (int, float)) else None


def assert_pca_pair(
    ckpt_pca: Mapping[str, Any] | None,
    cache_pca: Mapping[str, Any] | None,
) -> None:
    """Refuse silent PCA fallback across a different basis."""
    if not ckpt_pca or not cache_pca:
        return
    ckpt_keys = set(ckpt_pca)
    cache_keys = set(cache_pca)
    if ckpt_keys != cache_keys:
        raise ValueError(
            f"checkpoint PCA keys {sorted(ckpt_keys)} != cache PCA keys {sorted(cache_keys)}. "
            "Never mix checkpoint PCA with a different cache."
        )
    for name in ckpt_keys:
        cn, bn = _pca_n_components(ckpt_pca[name]), _pca_n_components(cache_pca[name])
        if cn is not None and bn is not None and cn != bn:
            raise ValueError(
                f"PCA n_components mismatch for {name}: checkpoint={cn} cache={bn}. "
                "Never mix checkpoint PCA with a different cache."
            )


def assert_cache_checkpoint_pair(
    config_tag: str | None,
    cache_tag: str | None,
    ckpt_pca: Mapping[str, Any] | None,
    cache_pca: Mapping[str, Any] | None,
) -> None:
    assert_dataset_tags(config_tag, cache_tag)
    assert_pca_pair(ckpt_pca, cache_pca)
