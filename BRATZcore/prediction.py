import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from tqdm import tqdm
import argparse

from .model import UNet
from .configuration import DRIVE_PATH
from .utils import compute_segmentation_metrics


class Patient_Database:
    def __init__(self, path):
        """Initialize database: locate patient folders and set compute device."""
        # base paths for data splits
        self.overall_path = path
        self.dataset_base = Path(path) / 'data'
        self.training_base = self.dataset_base / "training"
        self.test_base = self.dataset_base / "test"

        # list all patient folders under training and test
        self.training_patient_folders = sorted(self.training_base.glob('patient???'))
        self.test_patient_folders     = sorted(self.test_base.glob('patient???'))

        # report counts
        print(f"Found {len(self.training_patient_folders)} training patients,",
              f"{len(self.test_patient_folders)} test patients.")

        # choose GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # gather all patient IDs (assumes method defined elsewhere)
        self.allPIDs = self.get_all_patient_ids()

    def setup_model(self, modelpath, run_name=None):
        """Load saved UNet model and metadata, reconstruct network on correct device."""
        # build checkpoint path and remember it
        ckpt_path = os.path.join(self.overall_path, 'Models', modelpath)
        self.modelpath = modelpath

        # load model state dict
        sd = torch.load(ckpt_path, map_location=self.device)

        # load associated metadata (features list, run name)
        metadata_path = os.path.splitext(ckpt_path)[0] + '_metadata.npz'
        metadata = np.load(metadata_path, allow_pickle=True)
        features = metadata['features'].tolist()
        # decide run_name: argument overrides metadata
        self.run_name = run_name or metadata['run_name'].item()

        # reconstruct UNet with same in/out channels and feature map sizes
        in_ch  = 1  # MRI inputs are single-channel
        out_ch = sd['final.weight'].shape[0]  # number of output classes
        self.model = UNet(in_ch, out_ch, features).to(self.device)

        # load weights and set to evaluation mode
        self.model.load_state_dict(sd)
        self.model.eval()

    def get_all_patient_ids(self):
        """Return sorted list of all patient IDs from training and test."""
        # extract numeric IDs from folder names
        training_ids = [int(p.name[-3:]) for p in self.training_patient_folders]
        test_ids     = [int(p.name[-3:]) for p in self.test_patient_folders]
        # combine and sort all IDs
        return sorted(training_ids + test_ids)
        
    def patient_file_loc(self, patient_id):
        """Get folder path for a given patient ID."""
        # choose training or test subfolder based on ID threshold
        if patient_id <= 100:
            # format ID with leading zeros
            folder = f"{self.overall_path}data/training/patient{patient_id:03d}/"
        else:
            folder = f"{self.overall_path}data/test/patient{patient_id:03d}/"
        return folder
        
    def predict(self, mri_volume: np.ndarray) -> np.ndarray:
        """Run the UNet on a 3D MRI volume to get per-slice masks."""
        # convert to float32 and scale to [0,1]
        norm_volume = mri_volume.astype(np.float32)
        norm_volume /= np.max(norm_volume)

        # make tensor of shape (D,1,H,W)
        slice_tensor = torch.from_numpy(norm_volume).unsqueeze(1)
        # resize slices to model input size
        resized_tensor = F.interpolate(
            slice_tensor.float(),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).to(self.device)

        # forward pass in batch
        with torch.no_grad():
            pred = self.model(resized_tensor)           # (D, C, 128, 128)
            pred_mask = torch.argmax(pred, dim=1)       # (D, 128, 128), on CPU

        return pred_mask.cpu().numpy()
    
    def predict_patient(self, patient_id: int):
        """Process all MRI volumes for one patient and save predicted masks."""
        folder = self.patient_file_loc(patient_id)
        files = os.listdir(folder)
        for fname in files:
            # skip already predicted files
            if "predicted" in fname or "metrics" in fname:
                continue

            file_path = os.path.join(folder, fname)
            # load MRI data (assumed stored in first array)
            with np.load(file_path, allow_pickle=True) as data:
                # get volume with shape (D, H, W)
                img_key, gt_key = data.files[0], data.files[1]
                mri_vol = data[img_key].transpose(2, 0, 1)   # (D,H,W)
                gt_vol  = data[gt_key].transpose(2, 0, 1)    # (D,H,W)

            # predict mask volume
            pred_small = self.predict(mri_vol)            # shape: (D, 128, 128)

            # 2) upsample prediction back to (H, W)
            D, H_gt, W_gt = gt_vol.shape
            pred_t   = torch.from_numpy(pred_small).unsqueeze(1).float()  # (D,1,128,128)
            pred_full = F.interpolate(
                pred_t,
                size=(H_gt, W_gt),
                mode='nearest'
            ).squeeze(1).numpy().astype(pred_small.dtype)                # (D, H, W)

            # 3) compute metrics at full res
            metrics = compute_segmentation_metrics(gt_vol, pred_full)
            mname   = f"{fname[:-4]}_metrics_{self.run_name}.npz"
            np.savez(os.path.join(folder, mname), **metrics)

            # 4) save upsampled prediction for viewer (transpose back to H,W,D)
            pname = f"{fname[:-4]}_predicted_{self.run_name}.npz"
            np.savez(
                os.path.join(folder, pname),
                predicted_label=pred_full.transpose(1, 2, 0)
            )
                    
    def find_run_names(self):
        """Return unique run names from prediction files."""
        run_names = set() # collect unique names
        # scan all training and test folders
        for patient_folder in self.training_patient_folders + self.test_patient_folders:
                for fname in os.listdir(patient_folder):
                    if 'predicted' in fname:
                        # split off the run name after the last underscore, drop extension
                        run_name = fname.rsplit("_", 1)[-1].replace(".npz", "")
                        run_names.add(run_name)
        self.run_names = run_names # store for later use
        return run_names
    
    def delete_predictions(self, run_name):
        """Delete all prediction files matching a run name after confirmation."""
        n_found =0
        # first, list and count files to delete
        for patient_folder in self.training_patient_folders + self.test_patient_folders:
                for fname in os.listdir(patient_folder):
                    if 'predicted' in fname and run_name in fname:
                        print(fname) # show candidate 
                        n_found += 1

        # confirm deletion
        confirm = input(f"Are you sure you want to delete all {n_found} predictions for {run_name}? (y/n) ")
        if confirm.lower() == "y":
            # remove each matching file
            for patient_folder in self.training_patient_folders + self.test_patient_folders:
                for fname in os.listdir(patient_folder):
                    if 'predicted' in fname and run_name in fname:
                        file_to_remove = os.path.join(patient_folder, fname)
                        os.remove(file_to_remove) # delete file
                        print(f"Removed: {file_to_remove}")

    def load_patient_data(self, folder: str, frame: str, pred_path: str):
        """Load MRI volume, ground truth, and predicted masks from files."""
        # open the MRI file and corresponding predictions
        with np.load(os.path.join(folder, frame)) as data, np.load(pred_path) as pred_data:
            keys = data.files
            # assume first array is MRI volume, second is GT mask
            mri = data[keys[0]]
            label = data[keys[1]]
            # predictions stored under 'predicted_label'
            pred = pred_data['predicted_label']

            # convert from (H, W, D) to (D, H, W) for all volumes
            mri = np.transpose(mri, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            pred = np.transpose(pred, (2, 0, 1))
        
            # Return all slices for further selection/plotting
            return mri, label, pred
        
    def reshape_slices(self, mri, label, pred):
        """Ensure label and prediction masks match each MRI slice’s shape.

        Args:
            mri (np.ndarray): MRI volume (D, H, W).
            label (np.ndarray): Ground Truth (GT) masks (D, H?, W?).
            pred (np.ndarray): Predicted masks (D, H?, W?).

        Returns:
            tuple: (mri, label_aligned, pred_aligned) as np.ndarrays.
        """
        label_aligned = []
        pred_aligned = []

        # iterate over each slice index
        for i in range(len(mri)):
            # extract corresponding slices
            mri_slice = mri[i]      
            label_slice = label[i]    
            pred_slice = pred[i]

            # if ground-truth mask size differs, resize to match MRI slice
            if label_slice.shape != mri_slice.shape:
                label_slice = F.interpolate(
                    torch.from_numpy(label_slice)
                         .unsqueeze(0).unsqueeze(0)
                         .float(),
                    size=mri_slice.shape,
                    mode='nearest'
                ).squeeze().numpy().astype(np.uint8)

            # if prediction mask size differs, resize to match MRI slice
            if pred_slice.shape != mri_slice.shape:
                pred_slice = F.interpolate(
                    torch.from_numpy(pred_slice)
                         .unsqueeze(0).unsqueeze(0)
                         .float(),
                    size=mri_slice.shape,
                    mode='nearest'
                ).squeeze().numpy().astype(np.uint8)

            label_aligned.append(label_slice)
            pred_aligned.append(pred_slice)

        # stack back into volumes
        return mri, np.array(label_aligned), np.array(pred_aligned)
    
    
    def plot_slice(self, mri, label, pred, slice_no, show=False):
        """Plot one slice with ground truth vs. prediction overlays.

        Args:
            mri (np.ndarray): Volume as (D, H, W).
            label (np.ndarray): GT masks (D, H, W).
            pred (np.ndarray): Predicted masks (D, H, W).
            slice_no (int): Index of slice to plot.
            show (bool): If True, display plot; otherwise close it.

        Returns:
            tuple: (mri, label_overlay, pred_overlay) arrays.
        """
        # define semi-transparent colors for classes 1–3
        overlay_cmap = ListedColormap([
            [0.8, 0.2, 0.2],  # class 1: red
            [0.2, 0.8, 0.2],  # class 2: green
            [0.2, 0.2, 0.8],  # class 3: blue
        ])

        # mask out background (0) so only classes show in overlay
        label_overlay = np.ma.masked_where(label == 0, label)
        pred_overlay  = np.ma.masked_where(pred == 0, pred)

        i = slice_no  # alias for readability
        mri_slice   = mri[i]           # gray image
        label_slice = label_overlay[i] # masked GT
        pred_slice  = pred_overlay[i]  # masked prediction

        # set up side-by-side subplots
        fig, axs = plt.subplots(1, 2, figsize=(5, 5))

        # left: ground truth overlay
        axs[0].imshow(mri_slice, cmap='gray')
        axs[0].imshow(label_slice, cmap=overlay_cmap, alpha=0.4, vmin=1, vmax=3)
        axs[0].set_title(f"Ground Truth | Slice {i}")
        axs[0].axis('off')
        axs[0].set_aspect('equal')

        # right: prediction overlay
        axs[1].imshow(mri_slice, cmap='gray')
        axs[1].imshow(pred_slice, cmap=overlay_cmap, alpha=0.4, vmin=1, vmax=3)
        axs[1].set_title(f"Prediction   | Slice {i}")
        axs[1].axis('off')
        axs[1].set_aspect('equal')

        plt.tight_layout()
        self.fig = fig  # store figure on object for later use

        if show:
            plt.show()   # display immediately
        else:
            plt.close()  # keep headless

        return mri, label_overlay, pred_overlay 
    
    def plot_slices(self, mri, label, pred, max_slices=4, show=False):
        """Plot up to max_slices evenly spaced slices."""
        # decide how many slices to display
        n_slices = min(mri.shape[0], max_slices)
        # pick evenly spaced slice indices
        slice_indices = np.linspace(0, mri.shape[0] - 1, n_slices, dtype=int)
        # plot each selected slice
        for i in slice_indices:
            self.plot_slice(mri, label, pred, i, show=show)
        # return arrays for possible further use
        return mri, label, pred
    
    def plot_pred(self, mri, label, pred, max_slices=4, show=False):
        """Alias for plot_slices to display predictions."""
        # determine count and indices as above
        n_slices = min(mri.shape[0], max_slices)
        slice_indices = np.linspace(0, mri.shape[0] - 1, n_slices, dtype=int)
        # reuse plot_slice for each index
        for i in slice_indices:
            self.plot_slice(mri, label, pred, i, show=show)
        return mri, label, pred
        
    def iterate_patient_files(self, patient_id, funcs, run_name=None, frame=None):
        """
        Loop over a patient’s frames, loading data and applying funcs pipeline.

        Args:
            patient_id (int): ID of the patient.
            funcs (list of tuple): [(func, kwargs), …] to apply to (mri, label, pred).
            run_name (str, optional): Override run name for predictions.
            frame (str, optional): Specific .npz frame to process; defaults to all.
        """
        # use provided run_name or default
        if run_name is None:
            run_name = self.run_name

        # locate patient folder
        folder = self.patient_file_loc(patient_id)

        # gather target frame files
        if frame is None:
            # all original .npz files except predictions and hidden files
            frame_files = sorted(
                f for f in os.listdir(folder)
                if f.endswith('.npz') and 'predicted' not in f and not f.startswith('.')
            )
        else:
            frame_files = [frame]

        # process each frame
        for frame_file in frame_files:
            base = frame_file[:-4]  # strip ".npz"
            pred_file = f"{base}_predicted_{run_name}.npz"
            pred_path = os.path.join(folder, pred_file)

            # skip if prediction missing
            if not os.path.exists(pred_path):
                print(f"Prediction missing for {frame_file}, skipping.")
                continue

            # load MRI, ground truth, and prediction volumes
            mri, label, pred = self.load_patient_data(folder, frame_file, pred_path)

            # apply each function in the pipeline
            for func, kwargs in funcs:
                mri, label, pred = func(mri, label, pred, **kwargs)
            

if __name__ == "__main__":
    # parse command-line arguments for model file and run name
    parser = argparse.ArgumentParser(description="Run prediction on BRATZ dataset")
    parser.add_argument(
        "--modelpath", type=str, required=True,
        help="Model checkpoint path under Models/"
    )
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="Identifier for this prediction run (no underscores)"
    )
    args = parser.parse_args()

    # ensure run_name has no underscores to avoid filename conflicts
    if "_" in args.run_name:
        raise ValueError(
            "run_name must not contain underscores ('_'). Please choose a different name."
        )

    # initialize database and load the specified model
    pd = Patient_Database(path=DRIVE_PATH)
    pd.setup_model(modelpath=args.modelpath, run_name=args.run_name)

    # run prediction for every patient ID in the database
    for pid in tqdm(pd.allPIDs, desc="Predicting patients"):
        pd.predict_patient(patient_id=pid)

# python -m BRATZcore.prediction --modelpath unet_model.pth --run_name untesty
# python -m BRATZcore.prediction --modelpath unet_model_05Jul_19:28.pth --run_name no_augmentationv3