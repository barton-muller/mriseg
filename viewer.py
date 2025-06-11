# Local imports
from BRATZcore.configuration import DRIVE_PATH
from BRATZcore.prediction import Patient_Database
from BRATZcore.utils import calculate_segment_volumes

# General libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas

### INSTRUCTIONS ###
# In terminal run this command:
# streamlit run viewer.py

# Set the page title
st.title("BRATZ Patient Prediction Viewer")

# Base path for all models and data
drive_path = DRIVE_PATH

# --- Model selection ---
models_dir = os.path.join(drive_path, "Models")
# List all .pth files in the models directory
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
if not model_files:
    # If no models found, show an error and stop the app
    st.error("No model files found in Models/. Please add a .pth model file.")
    st.stop()

# Dropdown for selecting which model to view
modelpath = st.selectbox("Select model", model_files)
modelpath_full = modelpath
# Strip extension to form the base path for metadata files
metadata_base  = os.path.splitext(os.path.join(models_dir, modelpath_full))[0]

# --- Training metrics section ---
# Checkbox to toggle display of validation metrics over epochs
if st.checkbox("Show validation metrics over epochs"):
    npz_path = metadata_base + "_metadata.npz"
    if os.path.exists(npz_path):
        # Load recorded metrics from the .npz file
        data     = np.load(npz_path, allow_pickle=True)
        val_acc  = data.get("val_acc")
        val_dice = data.get("val_dice")
        val_prec = data.get("val_precision")
        val_rec  = data.get("val_recall")
        val_iou  = data.get("val_iou")

        # Ensure the essential arrays are present
        if val_acc is not None:
            st.subheader("Validation Metrics over Epochs")
            
            # Build dict of metrics to plot
            metrics = {"Accuracy": val_acc}
            # Conditionally add the new metrics if they exist
            if val_dice is not None:
                metrics["Dice (F1)"] = val_dice
            if val_prec is not None:
                metrics["Precision"] = val_prec
            if val_rec is not None:
                metrics["Recall"] = val_rec
            if val_iou is not None:
                metrics["IoU"] = val_iou

            # Render interactive line chart
            st.line_chart(metrics)
        else:
            # Warn user if essential keys are missing
            st.warning("Missing required keys (‘val_acc’ or ‘val_dice’) in metadata .npz")
    else:
        st.info(f"No metadata file found at {npz_path!r}. Run training to generate it.")

# --- Plot Loss function over epochs (optional) ---
# Checkbox to toggle training vs. validation loss plot
if st.checkbox("Show training and validation loss"):
    metadata_base = os.path.join(models_dir, os.path.splitext(modelpath_full)[0])
    try:
        # Load train/test loss arrays
        data = np.load(metadata_base + "_metadata.npz", allow_pickle=True)
        train_loss = data["train_loss"]
        test_loss  = data["val_loss"]

        loss_df = pandas.DataFrame({
            "Train Loss": train_loss,
            "Validation Loss": test_loss
        })

        st.subheader("Training vs. Validation Loss per Epoch")
        st.line_chart(loss_df)

    except FileNotFoundError:
        # Warn if metadata file missing
        st.warning(f"Metadata file not found at {metadata_base}.npz")

# Initialize the patient database interface
pd = Patient_Database(drive_path)

# --- Run name selection ---
# Gather all run names from the database
run_names = pd.find_run_names()
default_run_name = os.path.splitext(modelpath)[0]
use_new_run_name = st.checkbox("Enter new run name")

# Allow user to enter a custom run name or select from existing ones
if use_new_run_name:
    run_name = st.text_input("New run name", default_run_name)
else:
    run_name = st.selectbox("Run name", sorted(run_names))

# Configure the model and run name in the database handler
pd.setup_model(modelpath=modelpath_full, run_name=run_name)

# --- Patient selection ---
# Extract numeric patient IDs from folder names
test_patients     = [int(str(p).split('patient')[-1]) for p in pd.test_patient_folders]
training_patients = [int(str(p).split('patient')[-1]) for p in pd.training_patient_folders]
all_patients      = training_patients + test_patients

# Dropdown to select which patient to view
patient_id = st.selectbox("Select patient", sorted(all_patients))

# --- Frame selection ---
# Locate the folder for the selected patient
folder = pd.patient_file_loc(patient_id)
# List all MRI frame files (exclude predicted ones and hidden files)
frame_files = sorted([f for f in os.listdir(folder) if f.endswith('.npz') and 'predicted' not in f and not f.startswith('.')])
frame = st.selectbox("Select frame", frame_files)

# --- Prediction ---
# Determine the expected prediction filename
pred_file = frame.replace('.npz', f'_predicted_{pd.run_name}.npz')
pred_path = os.path.join(folder, pred_file)

if not os.path.exists(pred_path):
    st.warning("No prediction yet for this frame.")
    # If no prediction exists yet, show button to run prediction
    if st.button("Predict for this frame"):
        pd.predict_patient(patient_id)
        st.success("Prediction complete. Please re-select the frame.")
        st.stop()
else:
    # Inform user that a prediction is available
    st.info(f"Prediction exists for this frame {frame}")

    # Load the raw MRI, ground truth label, and model prediction
    mri, label, pred = pd.load_patient_data(folder, frame, pred_path)

    # Build a small pipeline of functions to reshape and plot a slice
    n_slices = mri.shape[0]
    slice_idx = st.slider("Slice", 0, n_slices-1, 0)

    func_pipeline = [
        (pd.reshape_slices, {}),
        (pd.plot_slice, {'slice_no': slice_idx, 'show': False})
    ]
    # Apply the pipeline and draw the figure
    pd.iterate_patient_files(patient_id, func_pipeline, frame=frame)
    st.pyplot(pd.fig)

    # — now display the test‐set metrics for this frame —
    metrics_file = frame.replace('.npz', f'_metrics_{pd.run_name}.npz')
    metrics_path = os.path.join(folder, metrics_file)
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True)
        st.subheader("Prediction Metrics")

        formatted_metrics = {
            name.replace('_', ' ').title(): float(val)
            for name, val in metrics.items()
        }

        metrics_df = pandas.DataFrame.from_dict(formatted_metrics, orient='index', columns=['Value'])
        metrics_df['Value'] = metrics_df['Value'].clip(upper=1.0)

        # Display as bar chart
        st.bar_chart(metrics_df)
    else:
        st.warning("No metrics found for this frame. Run prediction with metrics enabled.")


# --- Volume calculation ---
# Compute volume of each segmented region (in cm³)
voxel_volume = 1.3 * 1.3 * 10 / 1000  # Convert mm³ to cm³

# Compute predicted and ground truth volumes
pred_volumes = calculate_segment_volumes(pred, voxel_volume)
gt_volumes   = calculate_segment_volumes(label, voxel_volume)

# Display computed volumes and % error
st.subheader("Predicted Anatomical Volumes with Error")

segment_names = {
    1: "Right Ventricle",
    2: "Myocardium",
    3: "Left Ventricle"
}

for segment in sorted(pred_volumes):
    pred_vol = pred_volumes[segment]
    gt_vol = gt_volumes[segment]
    name = segment_names.get(segment, f"Segment {segment}")

    # Calculate % error safely
    if gt_vol > 0:
        percent_error = 100 * abs(pred_vol - gt_vol) / gt_vol
        st.write(f"{name}: {pred_vol:.2f} cm³  (Error: {percent_error:.1f}%)")
    else:
        st.write(f"{name}: {pred_vol:.2f} cm³  (Ground truth volume = 0)")