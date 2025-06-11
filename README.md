# Image Segmentation with Machine Learning

In this project we explore the utilisation of Unet model to segment MRI images. 

## Current basic folder structure:
- repository:
    - BRAZTcore/
      - __init__.py
      - configuration.py
      - utils.py
      - model.py
      - training.py
      - prediction.py
    - .gitignore
    - README.md
    - plotting_presets.py
    - requirements.txt
    - viewer.py
  
- data and model files:
  - data/
    - test/
      - PatientXXX/
        - patientXXX_frameYY.npz
        - patientXXX_frameYY_predicted_runname.npz
    - training/
      - PatientXXX/
        - patientXXX_frameYY.npz
        - patientXXX_frameYY_predicted_runname.npz
  - Models/
    - model.pth files
    - unet_model.pth (ROSAv1)




## Files in BRATZcore

### configuration.py

It contains path to data and model files `DRIVE_PATH = '.../BRATZ/`.

It will be added to gitignore to ensure your own path is not pushed to github.

### utils.py

This file consists of many definitions for functions encoding for: 
- Data Preprocessing & Transformation 
  - pad_crop_tensor() **Transforms a tensor to a desired shape by centrally cropping the exact desired shape**
  - smart_resize() **Converts images and segmentation masks to tensors**
  - normalize_image() **Normalizes pixels values**
- Metrics & Loss functions
  - dice_loss_function() **Calculates set similarity for each specified class**
  - get_dice_loss() **Partial Dice Loss function for specified categories**
  - make_loss_fn() **Supports multiple loss functions based on user's arguments**
- System Utilizations: Select Device() **Selects based available computing device**
- Data Loading
  - get_training_data() **Loads images and segmentation labels**
- Augmentation 
  - random_horizontal_flip() **Randomly flipping an image**
  - random_rotation() **Randomly rotates an image with an angle in the specified constraints**
  - random_resized_crop() **Randomly resize both the image and the segmentation label**
  - brightness_jitter() **Random jitter to image**



### model.py

This files contains the Unet model we incorporated in our project, based on Ronnenberg et al. (2025).

<img src="figs/Architecture_simplified.png" alt="streamlit_demo" width="300">


| Phase      | Operation                     | Purpose                             |
| ---------- | ----------------------------- | ----------------------------------- |
| Input      | Padding                       | Ensure size compatibility           |
| Encoding   | Conv => BN => ReLU => Pool    | Feature extraction + downsampling   |
| Bottleneck | Deeper convolutional layer    | Capture deep context                |
| Decoding   | Upsample=> Concat skip=> Conv | Restore resolution with fine detail |
| Output     | 1×1 Conv & Crop               | Produce segmentation map            |


### training.py

Contains `train` function for training a model on the training data aswell as entire pipline for the training process (maybe move elsewhere?)
Can be run with:
``` python -m BRATZcore.training --epochs 10 ```
There is a check before you start training and one after training is done, to make sure you don't overwrite existing files but this could result in loosing your train model if you mistype.
One can then view model with with viewer.py.

### prediction.py

Contains `Patient_Database` class for accessing patient data, making predictions based on a model, and visualizing predictions.

- Initialization & Setup
  - __init__() **Path for training and data sets, initialises patient databse**
  - setup_model() **Setup for the model, with Unet**

- Data Access & Management
  - get_all_patient_ids() **Sorted list of patient IDs**
  - patient_file_loc() **Path for a patient folder based on ID**
  - find_run_names() **Finds and returns all unique run names from prediction file names in the dataset**
  - delete_predictions() **Delete predictions for a specific run name**
  - iterate_patient_files() **Iterate through all patient files and apply the functions in a pipeline**

- Prediction
  - predict() **Predicted segmentation masks as a 3D numpy array**
  - predict_patient() **Predicted volume sizes in MRI**

- Data Loading
  - load_patient_data() **Extracts image, prediction and true labels**

- Postprocessing
  - reshape_slices() **Checks if label or prediction slices match the MRI dimensions**

- Visualization
  - plot_slice() **Truth vs prediction for one slice**
  - plot_slices() **Truth vs predicition for multiple slices**


#### Running predictions from the command line

```bash
python -m BRATZcore.prediction --modelpath unet_model.pth --run_name run_name
```

This iterates through all patients (test and training) and predicts the output for each using the specified model.
Predictions are saved in each patient's folder as:
```
patientXXX_frameYY_predicted_runname.npz
```

For the `ROSAv1` model, this process takes approximately 5 minutes on a MacBook M2.

The `run_name` is used to label the predictions so that different model outputs can be compared.

A custom pipeline can be implemented during plotting to support different visualizations or pre/post-processing.



## Viewer.py
This pyplot was created to help users in navigating through the results of the models. 
They allow selecting the model, a specific patient, and viewing the model performance metrics (F1 value and accuracy). 

#### Visualizing Predictions

The output files are best visualized with `viewer.py` using the Streamlit library. To launch the viewer:

```bash
streamlit run viewer.py
```

This opens an interactive Streamlit app that allows visualization of the predictions and will also compute predictions for new patients if a prediction file is not found.

<img src="figs/streamlit_demo.png" alt="streamlit_demo" width="300">

## Model Comparison

| Model Name     | File Name                                                     | Training Set Size |Augmentation                   | F1 Score      | Accuracy (Train/Test)       | Notes                             |
|----------------|---------------------------------------------------------------|-------------------|-------------------------------|---------------|-----------------------------|-----------------------------------|
| Full model     | 'full_unet_model_30epochs-full-dice.pth'                      | 80%               | hflip, rot10, resize.1, jit.2 |0.89           |99.22%                       |We used 30 epochs                  |
| Small model 30 | 'small_unet_model_30epochs-small-dice.pth'`                   | 80%               | hflip, rot10, resize.1, jit.2 |0.875          |99.12%                       |We used 30 epochs                  |
| Small model 50 | 'small_unet_model_50epochs-small-dice-20percent-data.pth'`    | 20%               | hflip, rot10, resize.1, jit.2 |0.836          |98.87%                       |We used 50 epochs                  |
| Small model 30 | 'small_unet_model_30epochs-small-dice-missing-predict.pth'`   | 80%               | hflip, rot10, resize.1, jit.2 |0.88           |99.09%                       |No prediction for class demo       |


## References
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Lecture Notes in Computer Science, 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28


# Tasks per member:
- Rosa: 
  - initial research into different models
  - implemented Unet model
  - code organisation
  - trained models initially and retrieved initial predictions
  - made Unet smaller
  - added volume prediction
  - presentation slides
  - presentation speaker
- Theo: 
  - made a version of the basic code
  - added code so it can be run from GPU
  - trained the final models
  - commented the code
  - made requirements.text
  - added docstrings
  - solved padding and cropping to the Unet model
  - presentation slides
  - presentation speaker
- Barton: 
  - made a version of the basic code
  - data augmentation
  - organised into pyplots
  - made the core functions folder
  - developed Streamlit GUI interface
  - organisation of ToDos
- Alin: 
  - modified one of Rosa's implementation so that it can be run from the terminal  
  - made the READme file
  - architecture diagram for our Unet model 
  - presentation slides
  - presentation speaker
- Zhi: 
  - added the Dice Loss function
  - added other metrics including TP, FP, FN, TN
  - added f1 function to predictions.py
