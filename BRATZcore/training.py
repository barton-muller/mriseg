# General libraries
import argparse
import sys
import os
import time
import inspect
from tqdm import tqdm
import numpy as np
from PIL import Image

# Machine learning related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

# Local imports
from .utils         import make_loss_fn, random_horizontal_flip, random_resized_crop
from .utils         import random_rotation, brightness_jitter, device, get_training_data
from .configuration import DRIVE_PATH
from .model         import UNet

class CardiacMRIDataset(Dataset):
    def __init__(self, training_img, training_label, train=True):
        """Dataset for cardiac MRI: images (N,1,H,W) and one-hot labels (N,4,H,W)."""
        # stack and add channel dim to images
        X_train = torch.Tensor(training_img).unsqueeze(1)
        # load labels and convert to long
        y = torch.from_numpy(training_label).long()
        print("Label has size:", y.size(), "Categories:", torch.unique(y))
        # one-hot encode labels to shape (N,4,H,W)
        y_onehot = torch.nn.functional.one_hot(y, num_classes=4).permute(0, 3, 1, 2).float()

        self.data = X_train
        self.labels = y_onehot
        self.train = train

    def __len__(self):
        """Returns the size of the dataset (number of samples)."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return (image, label); apply augmentations if in training mode."""
        data = self.data[idx]
        label = self.labels[idx]

        if self.train:
            # convert tensor to PIL for torchvision transforms
            img = transforms.ToPILImage()(data.squeeze(0))
            lbl = Image.fromarray(label.argmax(0).byte().numpy())

            # apply random flips, rotations, crops, and brightness jitter
            img, lbl = random_horizontal_flip(img, lbl)
            img, lbl = random_rotation(img, lbl)
            img, lbl = random_resized_crop(img, lbl)
            img = brightness_jitter(img)

            # convert back to tensor and one-hot encode label
            data = transforms.ToTensor()(img)
            lbl_arr = np.array(lbl, dtype=np.int64)
            label = torch.nn.functional.one_hot(
                torch.from_numpy(lbl_arr),
                num_classes=4
            ).permute(2, 0, 1).float()

        return data, label

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()  # set model to training mode (enables dropout, batchnorm update)
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    # iterate over batches
    for X, y in tqdm(dataloader, desc="Training", leave=False):
        # move inputs and labels to the target device (GPU/CPU)
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()   # clear previous gradients
        logits = model(X)       # forward pass: compute model outputs

        # compute loss: CE expects class indices, Dice expects one-hot
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # y.argmax(1) gives class index per pixel
            loss = loss_fn(logits, y.argmax(1))
        else:
            loss = loss_fn(logits, y)

        loss.backward()  # backpropagate gradients
        optimizer.step() # update model parameters

        total_loss += loss.item()             # accumulate loss
        preds = logits.argmax(1)              # predicted class per pixel
        total_correct += (preds == y.argmax(1)).float().sum().item()  # count correct pixels
        total_pixels += preds.numel()         # total number of pixels processed

    # compute average loss and accuracy over all batches
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels
    return avg_loss, accuracy


def validate_epoch(dataloader, model, loss_fn, device, categories=[1,2,3]):
    """Evaluate model. Returns evaluation metrics (avg_loss, accuracy, 
                                                    mean_iou, mean_dice or None)."""
    model.eval()  # set model to eval mode (disables dropout, fixes batchnorm)
    running_loss = 0.0
    correct = 0
    dice_scores = []
    # total pixels = number of samples * pixels per sample
    total_pixels = len(dataloader.dataset) * dataloader.dataset.data.shape[-1]**2

    # set up per-class counters for the evaluation metrics
    epsilon = 1e-8
    TP = {c: 0. for c in categories}
    FP = {c: 0. for c in categories}
    FN = {c: 0. for c in categories}

    # wrap in no_grad to save memory and computation
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validation", unit="batch", dynamic_ncols=True):
            # move to device
            X, y = X.to(device), y.to(device)
            logits = model(X)  # forward pass

            probs  = F.softmax(logits, dim=1)  # softmax to get probabilities

            # compute loss and, for Dice, record score
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(logits, y.argmax(1))
            else:
                loss = loss_fn(logits, y)
                # loss = 1 - dice_score, so record dice_score = 1 - loss
                dice_scores.append(1.0 - loss.item())

            running_loss += loss.item()  # accumulate batch loss
            # count correct pixels
            correct += (logits.argmax(1) == y.argmax(1)).float().sum().item()

            # accumulate per‐class TP/FP/FN
            for c in categories:
                pred_c = probs[:, c]
                target_c = y[:, c]

                TP[c] += torch.sum(pred_c * target_c).item()
                FP[c] += torch.sum(pred_c * (1 - target_c)).item()
                FN[c] += torch.sum((1 - pred_c) * target_c).item()


    # average results
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total_pixels
    mean_dice = float(np.mean(dice_scores)) if dice_scores else None

    # compute macro metrics
    precisions = [TP[c] / (TP[c] + FP[c] + epsilon) for c in categories]
    recalls    = [TP[c] / (TP[c] + FN[c] + epsilon) for c in categories]
    f1s        = [2 * precisions[i] * recalls[i] /
                   (precisions[i] + recalls[i] + epsilon) for i in range(len(categories))]
    ious       = [TP[c] / (TP[c] + FP[c] + FN[c] + epsilon) for c in categories]

    macro_precision = np.mean(precisions)
    macro_recall    = np.mean(recalls)
    macro_f1        = np.mean(f1s)
    macro_iou       = np.mean(ious)

    return (avg_loss, accuracy, mean_dice, macro_precision, macro_recall, macro_f1, macro_iou)

def get_args():
    """Parse and return command-line arguments for training the segmentation model."""

    parser = argparse.ArgumentParser(description="Train segmentation model with flexible loss")

    # training parameters
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--model', type=str,         
                        choices=["full", "small"],
                        default="full",
                        help="Which U-Net to use: “full” for the full UNet, “small” for smallUNet")

    parser.add_argument('--loss', type=str,
                        choices=['cross_entropy', 'dice'],
                        default='cross_entropy',
                        help="Which loss to use: ('dice' or 'cross_entropy')")
    
    parser.add_argument('--categories', type=int, nargs='+',
                        default=[1,2,3],
                        help="(Dice only) list of channel indices to include")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")

    parser.add_argument("--run_name", type=str, 
                        default=None,
                        help="Short identifier for this training run. If not set, a timestamp will be used.")

    return parser.parse_args()

# Main execution of all of the above functions
if __name__ == "__main__":
    # Get all the training parameters (that were inputted into the parser as flags)
    args = get_args()

    # build model
    if args.model == "small":
        # small UNet:
        features=[6, 12, 25, 50]
        model = UNet(in_channels=1, out_channels=4,
                     features=features).to(device)
        print("Using small U-Net")
    else:
        # full UNet:
        features = [64, 128, 256, 512]
        model = UNet(in_channels=1, out_channels=4,
                     features=features).to(device)
        print("Using full U-Net (default features)")
    
    print(f"Using model: {args.model}")
    print(f"Training for {args.epochs} epochs with {args.loss} loss...\n")

    # Load and split data
    imgs, labels = get_training_data()
    N = len(imgs)
    idx = np.random.permutation(N)
    split = int(0.2 * N) # Hard coded that 80% of data is used for training and 20% for validation
    train_idx, val_idx = idx[:split], idx[split:]

    print(f"-> Total slices:      {N}")
    print(f"-> Training slices:   {len(train_idx)}")
    print(f"-> Validation slices: {len(val_idx)}\n")

    train_ds = CardiacMRIDataset(imgs[train_idx], labels[train_idx], train=True)
    val_ds   = CardiacMRIDataset(imgs[val_idx],   labels[val_idx],   train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # Loss & optimizer
    loss_fn  = make_loss_fn(args)
    optimizer= optim.Adam(model.parameters(), lr=args.lr)

    # Histories
    history_train_loss, history_train_acc = [], []
    history_val_loss,   history_val_acc   = [], []
    history_f1_scores                = []
    history_precision                = []
    history_recall                   = []
    history_iou                      = []

    # Build a base name for the model
    prefix     = "small" if args.model == "small" else "full"
    model_base = f"{prefix}_unet_model"

    # Prompt
    epochs = args.epochs
    if input(f"Train Model {model_base} for {epochs} epochs (y/n)? ") != "y":
        print("Aborted.")
        sys.exit()

    # Training loop
    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        # run one epoch
        ep_tr_loss, ep_tr_acc = train_epoch(train_loader, model, loss_fn, optimizer, device)
        (ep_val_loss,
        ep_val_acc,
        ep_val_dice,
        ep_val_precision,
        ep_val_recall,
        ep_val_f1,
        ep_val_iou) = validate_epoch(val_loader, model, loss_fn, device)

        # record into histories
        history_train_loss.append(ep_tr_loss)
        history_train_acc.append(ep_tr_acc)
        history_val_loss.append(ep_val_loss)
        history_val_acc.append(ep_val_acc)
        if ep_val_dice is not None:
            history_f1_scores.append(ep_val_dice)
            history_precision.append(ep_val_precision)
            history_recall.append(ep_val_recall)
            history_iou.append(ep_val_iou)

        # print
        print(f"Train -> loss: {ep_tr_loss:.4f}, acc: {100*ep_tr_acc:.1f}%")
        print(
            f" Val  -> loss: {ep_val_loss:.4f}, acc: {100*ep_val_acc:.1f}%"
            + (f", dice: {ep_val_dice:.4f}" if ep_val_dice is not None else "")
        )

    # Make sure Models directory exists
    models_dir = os.path.join(DRIVE_PATH, "Models")
    os.makedirs(models_dir, exist_ok=True)

    # Save weights
    timestamp = time.strftime("_%d%b_%H-%M")
    run_name = args.run_name or timestamp
    model_filename = f"{model_base}_{run_name}.pth"
    model_path     = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved weights to {model_path}")

    # Save metadata & history
    meta = {
        "train_loss": history_train_loss,
        "train_acc":  history_train_acc,
        "val_loss":   history_val_loss,
        "val_acc":    history_val_acc,
        "val_dice":   history_f1_scores,
        "val_precision": history_precision,
        "val_recall":    history_recall,
        "val_iou":       history_iou,
        "model":      args.model,
        "loss":       args.loss,
        "features":   features,
        "run_name":   run_name,
        "augmentation_fn": inspect.getsource(CardiacMRIDataset.__getitem__),
    }
    meta_filename = os.path.splitext(model_filename)[0] + "_metadata.npz"
    meta_path     = os.path.join(models_dir, meta_filename)
    np.savez(meta_path, **meta)
    print(f"Saved metadata to {meta_path}")

# python -m BRATZcore.training --epochs 1