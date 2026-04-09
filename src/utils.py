"""Feature extraction utilities for model benchmarking."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    transforms: object | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a dataloader.

    Args:
        model: Model to use for feature extraction.
        dataloader: DataLoader yielding dicts with ``"image"`` and ``"label"`` keys.
        device: Device to run inference on.
        transforms: Optional transform applied to images before the model.
        verbose: Whether to display a progress bar.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    x_all = []
    y_all = []

    enumerator = enumerate(dataloader)
    if verbose:
        enumerator = enumerate(tqdm(dataloader, total=len(dataloader)))

    for i, batch in enumerator:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()

        if transforms is not None:
            images = transforms(images)

        with torch.no_grad():
            with torch.inference_mode():
                features = model(images)
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                else:
                    if "norm" in features:
                        features = features["norm"].detach().cpu().numpy()
                    elif "global_pool" in features:
                        features = features["global_pool"].detach().cpu().numpy()
                    elif "head.global_pool" in features:
                        features = features["head.global_pool"].detach().cpu().numpy().squeeze()
                    else:
                        raise ValueError(f"Unexpected features format: {features.keys()}")

                # handles the case where features are 1D (e.g., the ResNet model has batch x features)
                if len(features.shape) == 1:
                    features = features[np.newaxis, :]

                # handles the case where features are 3D (e.g., the DinoV2 model has batch x tokens x features)
                if len(features.shape) == 3:
                    features = np.mean(features, axis=1, keepdims=False)

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def extract_features_transformers(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    processor: object,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features using a HuggingFace-style model with a processor.

    Args:
        model: Transformer model for feature extraction.
        dataloader: DataLoader yielding dicts with ``"image"`` and ``"label"`` keys.
        device: Device to run inference on.
        processor: Preprocessing callable applied to images before the model.
        verbose: Whether to display a progress bar.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    x_all = []
    y_all = []

    enumerator = enumerate(dataloader)
    if verbose:
        enumerator = enumerate(tqdm(dataloader, total=len(dataloader)))

    for i, batch in enumerator:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()

        images = processor(images)
        with torch.no_grad(), torch.inference_mode():
            features = model(images)

        x_all.append(features.cpu().numpy())
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all
