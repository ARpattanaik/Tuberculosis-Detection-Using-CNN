# Tuberculosis-Detection-Using-CNN
A deep-learning model that classifies chest X-ray images to detect signs of tuberculosis. Includes data preprocessing, CNN training, evaluation, and prediction pipeline for fast and reliable TB screening.


cat > README.md <<'EOF'
# TB Detection from Chest X-Ray

This repository contains a small TensorFlow/Keras pipeline to train a binary classifier that detects Tuberculosis (TB) from chest X-ray images.

Overview
- `TBdetection.py` - Main script. It downloads the Kaggle dataset (via `kagglehub`), prepares training and validation ImageDataGenerators, defines a compact CNN, trains the model, visualizes training curves, and saves the trained model as `tuberculosis_model.h5`.

Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- OpenCV (cv2)
- Pillow
- matplotlib
- seaborn
- kagglehub
- ipywidgets (optional, used in notebooks)

Install dependencies (recommended inside a virtual environment):

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow scikit-learn opencv-python pillow matplotlib seaborn kagglehub ipywidgets

Usage
1. Ensure you have access to Kaggle and configured credentials if needed by `kagglehub`. The script calls:
kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
to download the dataset.
2. Run the script from the directory that contains `TBdetection.py`:

python TBdetection.py

What the script does
- Downloads the TB chest X-ray dataset to the local Kaggle cache path used by `kagglehub`.
- Loads images from the dataset folders, sets up `ImageDataGenerator` for training/validation, and defines a simple CNN architecture.
- Trains the model for 10 epochs and saves the trained model to `tuberculosis_model.h5`.
- Plots training/validation accuracy and loss.

Notes and assumptions
- The script uses hard-coded dataset paths that point to the `kagglehub` cache (/root/.cache/kagglehub/...). If you run this on macOS or another environment, update `TBpath`, `Normalpath`, and `base_path` in `TBdetection.py` to match where the dataset is downloaded.
- The model and training pipeline are intentionally simple. For production or improved accuracy, consider using transfer learning (e.g., MobileNet, EfficientNet), image augmentation, class balancing, and proper hyperparameter tuning.

Saving and model output
- The trained model is saved as `tuberculosis_model.h5` in the script's working directory.

License
- Add a license file or specify your preferred license.

Contributing
- Improvements, bug fixes, and enhancements are welcome. Open issues or PRs with concrete suggestions.
EOF
