import numpy as np 
from pathlib import Path
from typing import Tuple
from functools import partial
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import random
from torchvision.transforms import functional as TF

from .configuration import DRIVE_PATH

def pad_crop_tensor(x: torch.Tensor, target_shape: Tuple[int, int]) -> \
        Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    """Pad or crop tensor to target spatial size.

    Args:
        x (Tensor): Input tensor of shape (B, C, H, W).
        target_shape (tuple): Desired (H', W').

    Returns:
        (Tensor, tuple, tuple): Transformed tensor, pad_info, crop_info.
    """
    # original height and width
    h0, w0 = x.shape[-2:]
    # target height and width
    h1, w1 = target_shape
    # compute difference between target and original
    h_diff, w_diff = h1 - h0, w1 - w0

    # helper to get symmetric pad sizes for one dimension
    def get_padding_size(diff: int) -> Tuple[int, int]:
        """Return (before, after) padding for given diff."""
        if diff > 0:
            # split padding evenly (remainder goes to 'after')
            pd = (diff // 2, diff - diff // 2)
        else:
            pd = (0, 0)
        return pd

    # compute pad for height and width
    h_pd, w_pd = get_padding_size(h_diff), get_padding_size(w_diff)
    # pad order for torch: (left, right, top, bottom)
    pad_info = (*w_pd, *h_pd)

    # apply reflect padding
    x_pad = torch.nn.functional.pad(x, pad_info, mode='reflect')

    # after padding, compute new size
    hp, wp = x_pad.shape[-2:]
    h_diff, w_diff = hp - h1, wp - w1
    assert h_diff >= 0 and w_diff >= 0, "Padding did not seem to work!"

    # compute crop amounts (before, after) for height and width
    h_crop = (h_diff // 2, h_diff - h_diff // 2)
    w_crop = (w_diff // 2, w_diff - w_diff // 2)
    crop_info = (*w_crop, *h_crop)

    # perform center crop
    x_crop = x_pad[..., 
                   h_crop[0]: hp - h_crop[1], 
                   w_crop[0]: wp - w_crop[1]]

    return x_crop, pad_info, crop_info

def smart_resize(im: np.ndarray, seg: np.ndarray, desired_hw: int = 128):
    """Crop and resize image and segmentation to a square of size desired_hw.

    Args:
        im (np.ndarray): Image array of shape (H, W, C).
        seg (np.ndarray): Segmentation map of shape (H, W).
        desired_hw (int): Target height and width.

    Returns:
        im_rescale (np.ndarray): Resized image (H', W', C).
        seg_rescale (np.ndarray): Resized segmentation (H', W').
    """
    # determine the largest centered square crop size
    hw = min(*im.shape[:2])

    with torch.no_grad():
        # convert inputs to tensors and ensure correct dtypes
        im, seg = torch.from_numpy(im), torch.from_numpy(seg).to(dtype=torch.int64)
        # move channels first: (H, W, C) → (C, H, W)
        im = torch.permute(im, (2, 0, 1))
        seg = torch.permute(seg, (2, 0, 1))

        # add batch dimension to image
        im = torch.unsqueeze(im, 1)
        # convert segmentation to one-hot with 4 classes and float type
        seg = torch.nn.functional.one_hot(seg, num_classes=4)
        seg = torch.permute(seg, (0, 3, 1, 2)).to(torch.float32)

        # center-crop (with padding if needed) to square of side hw
        im_crop, _, _ = pad_crop_tensor(im, (hw, hw))
        seg_crop, _, _ = pad_crop_tensor(seg, (hw, hw))

        # resize both tensors to desired_hw using bilinear interpolation
        im_rescale = torch.nn.functional.interpolate(im_crop,
                                                     size=(desired_hw, desired_hw),
                                                     align_corners=True,
                                                     mode='bilinear', antialias=True)
        seg_rescale = torch.nn.functional.interpolate(seg_crop, 
                                                      size=(desired_hw, desired_hw),
                                                      align_corners=True,
                                                      mode='bilinear', antialias=True)

        # convert one-hot back to class labels
        seg_rescale = torch.argmax(seg_rescale, dim=1)

        # move tensors back to CPU numpy and drop batch/channel dims
        im_rescale = im_rescale.detach().cpu().numpy().squeeze()
        seg_rescale = seg_rescale.detach().cpu().numpy().squeeze()

        # restore channel-last order for image and segmentation
        im_rescale = np.transpose(im_rescale, (1, 2, 0))
        seg_rescale = np.transpose(seg_rescale, (1, 2, 0))

    return im_rescale, seg_rescale.astype(np.uint8)


def normalize_image(im: np.ndarray, qvals=(1, 99)):
    """Normalize image to [0,1] using given percentile range.

    Args:
        im (np.ndarray): Input image.
        qvals (tuple): Low and high percentiles.
    Returns:
        np.ndarray: Clipped, normalized image.
    """
    # compute intensity bounds from percentiles
    im_amin, im_amax = np.percentile(im, qvals)
    # scale to [0,1]
    im = (im - im_amin) / (im_amax - im_amin)
    # clip values outside [0,1]
    im = np.clip(im, 0., 1.)
    return im      

def calculate_segment_volumes(segmentation: np.ndarray, voxel_volume: float):
    """Compute volumes for each nonzero label in a 3D segmentation.

    Args:
        segmentation (np.ndarray): 3D label array.
        voxel_volume (float): Volume per voxel.
    Returns:
        dict: {label: volume_cm3}
    """
    # find all labels present
    segment_labels = np.unique(segmentation)
    volumes = {}
    for label in segment_labels:
        # skip background
        if label == 0:
            continue
        # count voxels for this label
        count = np.sum(segmentation == label)
        # convert total voxels to cm³
        volumes[label] = count * voxel_volume * 1e-3
    return volumes

def dice_loss_function(logits, label, categories=(1, 2, 3)):
    """Compute the Dice loss between the predicted probabilities and true labels for the specified categories.

    Args:
        logits (Tensor): Raw network output, shape (N, C, H, W).
        label (Tensor): One-hot ground truth, same shape as logits.
        categories (tuple): Class indices to include.

    Returns:
        Tensor: Scalar Dice loss.
    """
    # ensure prediction and label shapes match
    assert logits.shape == label.shape, "Logits and label must match!"

    # convert logits to probabilities
    prob = torch.nn.functional.softmax(logits, dim=1)

    dice_scores = []
    for c in categories:
        # select class c probability and ground truth
        p = prob[:, c]
        g = label[:, c]
        # compute Dice numerator and denominator
        num = 2 * (p * g).sum()
        den = p.sum() + g.sum()
        # add smoothing to avoid division by zero
        dice_scores.append((num + 1e-5) / (den + 1e-5))

    # Dice loss = 1 - mean Dice score
    return 1.0 - torch.mean(torch.stack(dice_scores))

def get_dice_loss(categories=(1,2,3)):
    """Return a dice_loss_function partial with fixed categories (just for convenience)."""
    return partial(dice_loss_function, categories=tuple(categories))

def select_device():
    """Choose the best available PyTorch device (preference: CUDA > MPS > CPU)."""
    # Check for NVIDIA GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        device = torch.device("cuda:0")
        # flag if it's an RTX series
        is_rtx = "RTX" in name.upper()
        print(f"CUDA available: using {name!r} (RTX? {is_rtx})")
        return device

    # Check for Apple Silicon GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS available:", torch.backends.mps.is_available())
        print("MPS built:", torch.backends.mps.is_built())
        return torch.device("mps")

    # Fallback to CPU
    print("No GPU detected—falling back to CPU")
    return torch.device("cpu")

device = select_device()

def make_loss_fn(args):
    """
    Construct a PyTorch loss function based on the given arguments.

    Args:
    args : argparse.Namespace
        The parsed command-line arguments, expected to contain the following:
        - loss : str
            The type of loss function to use. Currently supported values are
            "cross_entropy" and "dice".
        - categories : tuple
            If using the "dice" loss, the categories for which to compute the
            Dice coefficient. Defaults to (1, 2, 3).

    Returns:
    torch.nn.Module
        An instance of the chosen loss function.

    Raises:
    ValueError
        If an unknown loss type is specified.
    """
    # Cross-entropy loss
    if args.loss == "cross_entropy":
        # Define class weights
        weights = torch.tensor([0.1, 1.0, 1.0, 1.0], device=device)
        return nn.CrossEntropyLoss(weight=weights)
    
    # Dice loss for specified categories
    if args.loss == "dice":
        # Wrap dice_loss_function with chosen categories
        return get_dice_loss(categories=tuple(args.categories))

    raise ValueError(f"Unknown loss: {args.loss}")

def get_training_data():
    """Load all training images and labels from data/images/training.

    Returns:
        training_img (np.ndarray): Array of shape (N, H, W).
        training_label (np.ndarray): Array of shape (N, H, W).
    """
    # Path to saved labels
    data_base = Path(DRIVE_PATH) / "data" / "images" / "training"

    # Get all patient folders
    patient_folders = sorted(data_base.glob("patient???"))

    # Lists to hold all slices
    training_img = []
    training_label = []

    for patient_folder in patient_folders:
        label_files = sorted(patient_folder.glob("*.npz"))

        for label_path in label_files:
            # Load segmentation label
            data = np.load(label_path)
            label = data["label"]  # shape: (128, 128)

            # Normalize label shape if needed (e.g., expand dims for channel)
            training_label.append(label)

            # Derive matching image path and load the image slice
            image_path = label_path.with_suffix(".png")
            image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
            training_img.append(image)
            

    # stack lists into (N, H, W) arrays
    training_img = np.stack(training_img)
    training_label = np.stack(training_label)

    print(f"Total slices: {len(training_img)}")
    print(f"Image shape: {training_img.shape}, Label shape: {training_label.shape}\n")

    return training_img, training_label


def random_horizontal_flip(img, lbl):
    """Randomly flip image and label horizontally (50% chance).

    Args:
        img (PIL.Image): Input image.
        lbl (PIL.Image): Corresponding label.
    Returns:
        (PIL.Image, PIL.Image): Possibly flipped pair.
    """
    # flip both image and label together
    if random.random() > 0.5:
        img = TF.hflip(img)
        lbl = TF.hflip(lbl)
    return img, lbl

def random_rotation(img, lbl, angle_range=(-10, 10)): 
    """Rotate image and label by a random angle in angle_range.

    Args:
        img (PIL.Image): Input image.
        lbl (PIL.Image): Corresponding label.
        angle_range (tuple): Min and max rotation angles.
    Returns:
        (PIL.Image, PIL.Image): Rotated pair.
    """
    # pick a random angle
    angle = random.uniform(*angle_range)
    # rotate image with bilinear, label with nearest to preserve classes
    img = TF.rotate(img, angle, fill=0, interpolation=TF.InterpolationMode.BILINEAR)
    lbl = TF.rotate(lbl, angle, fill=0, interpolation=TF.InterpolationMode.NEAREST)
    return img, lbl


def random_resized_crop(img, lbl, size=(128, 128), scale=(0.9, 1.1), ratio=(1.0, 1.0)):
    """Random crop and resize image and label.

    Args:
        img (PIL.Image): Input image.
        lbl (PIL.Image): Corresponding label.
        size (tuple): Output (H, W).
        scale (tuple): Range of crop size relative to original.
        ratio (tuple): Range of aspect ratios.
    Returns:
        (PIL.Image, PIL.Image): Cropped and resized pair.
    """
    # get crop parameters
    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
    # apply same crop to image and label
    img = TF.resized_crop(img, i, j, h, w, size=size, interpolation=TF.InterpolationMode.BILINEAR)
    lbl = TF.resized_crop(lbl, i, j, h, w, size=size, interpolation=TF.InterpolationMode.NEAREST)
    return img, lbl


def brightness_jitter(img, brightness=0.2):
    """Apply random brightness adjustment to an image.

    Args:
        img (PIL.Image): Input image.
        brightness (float): Max brightness change factor.
    Returns:
        PIL.Image: Brightness-jittered image.
    """
    # use torchvision's ColorJitter for brightness
    return transforms.ColorJitter(brightness=brightness)(img)

def compute_segmentation_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute segmentation evaluation metrics between ground truth and prediction.

    Metrics include:
    - Pixel-wise accuracy
    - Per-class Dice coefficient
    - Per-class Intersection over Union (IoU)

    Args:
        gt (np.ndarray): Ground truth labels, shape (D, H, W)
        pred (np.ndarray): Predicted labels, shape (D, H, W)

    Returns:
        dict: Dictionary containing overall accuracy and class-wise Dice and IoU scores.
    """
    metrics = {}

    gt_flat = gt.ravel()
    pred_flat = pred.ravel()

    metrics['pixel_accuracy'] = float((pred_flat == gt_flat).sum() / gt_flat.size)

    # Compute class-wise Dice and IoU
    for cls in np.unique(gt_flat):
        if cls == 0:
            continue  # Optionally skip background class (label 0)

        gt_mask = (gt_flat == cls)
        pred_mask = (pred_flat == cls)

        tp = np.logical_and(gt_mask, pred_mask).sum() # True positives
        fp = np.logical_and(~gt_mask, pred_mask).sum() # False positives
        fn = np.logical_and(gt_mask, ~pred_mask).sum() # False negatives

        denom_dice = 2 * tp + fp + fn
        metrics[f'dice_class_{cls}'] = float(2 * tp / denom_dice) if denom_dice > 0 else np.nan

        denom_iou = tp + fp + fn
        metrics[f'iou_class_{cls}'] = float(tp / denom_iou) if denom_iou > 0 else np.nan

    return metrics