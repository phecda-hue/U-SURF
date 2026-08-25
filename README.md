# Photoacoustic Image Processing and Analysis

## 1. Public Datasets
1. VessMAP: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322048
2. Duke PAM dataset: https://zenodo.org/records/4042171
3. Mendeley data Photoacoustic vascular image dataset: https://data.mendeley.com/datasets/dp5jgrkd6k/2
4. UCL 3D Microvascular Image Data and Labels for Machine Learning: https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604?file=45996213


## 2. Predict Vessel Mask
Predicts blood vessel mask using U-Net and a Frangi filter.
The model consists of two input channels, which receive the original image and the image processed with the frangi filter, respectively.

### 1. train_models/unet_2ch_input/trainunet.py
RSOM data from the UCL dataset and Mendeley dataset were used.
predict_multicontrast.py predicts vessel mask using the model resulting from this code.
    
### 2. train_models/vessmap/train_vessmap.py
VessMAP dataset was used.
predict_vessmap_batch.py predicts vessel mask using the model resulting from this code.

### 3. combine_probability_sum.py
Creates a combined probability map from models resulted by trainunet.py and predict_vessmap.py.
The combined probabilities are filtered to generate the final vessel mask.


## 3. Centerline Correction
Identifies and corrects the curved line of symmetry in images of mouse ears.
This code extracts the centerline from a blood vessel mask predicted by U-Net.
It calculates the point along the x-axis where the blood vessel ratio is equal when the image is divided.


## 4. Denoising
Denoise the mouse ear and pig bile duct images.

### 1. mouse_ear(failed)/gan
trainmacnet.py and trainsrgan.py use phantom data generated from Duke PAM, Mendeley, UCL datasets.
Models resulted by these codes failed to denoise.

### 2. mouse_ear(failed)/distortion_correction
distortion_correction/bscan_demons_visualize.py uses harmonic fitting and demons algorithm.
This method corrects some noise, but takes too long time and still there are remaining noise.

### 3. Pig Bile Duct
Uses à trous wavelet transform.
Since the noise in the pig bile duct images varies in curvature and intensity, this code allows for the adjustment of parameters via a GUI and the accumulation of noise correction results.


## 5. Analyze Vessel
Using the generated blood vessel mask, the blood vessels are skeletonized to calculate parameters such as diameter, length, curvature, and cycle structure.
Calculations involving the skeleton are performed using sknw, with the skeleton represented as a graph for the computation.
When analyzing data, the exact width, height, and units (cm, mm, µm) for the scan range should be specified.