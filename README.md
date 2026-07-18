<div align="center">

# 😄 Real-Time Facial Emotion Recognition

### A custom CNN trained on cleaned FER-2013 data, with live browser-based emotion detection

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/TensorFlow-CNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow CNN" />
  <img src="https://img.shields.io/badge/Streamlit-Real--time-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-Face%20Detection-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
</p>

</div>

## ✨ Overview

This project recognises facial emotions from a live webcam feed. It uses a custom convolutional neural network (CNN) trained on a cleaned version of the FER-2013 dataset and predicts one of seven emotions:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

The project covers the full workflow: dataset-quality auditing, preprocessing, model training and evaluation, saved-model inference, and a real-time Streamlit application.

## 🚀 Run the real-time app

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Emotion-Prediction-using-CNN-model.git
cd Emotion-Prediction-using-CNN-model
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start real-time emotion detection 🎥

```bash
cd src
streamlit run run.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`) and allow browser camera access when prompted.

> [!IMPORTANT]
> Commit `Models/final_trained_model.keras` when you publish the repository. The application loads this bundled model automatically, using a path relative to the cloned project folder. No personal file path needs to be changed.

## 🧩 How the application is organised

`src/Inference.py` is the reusable inference module that supplies the essentials used by `src/run.py`:

- loading the trained Keras model;
- loading OpenCV's Haar cascade face detector;
- preprocessing each detected face to a `(1, 48, 48, 1)` tensor;
- defining emotion labels and colours; and
- drawing the predicted emotion, confidence, and probability bars.

`src/run.py` is the Streamlit + WebRTC entry point. It imports those essentials from `Inference.py`, receives browser webcam frames, and displays the annotated live video stream.

```text
Browser webcam → run.py (Streamlit/WebRTC) → Inference.py
                                            ├─ detect face
                                            ├─ preprocess to 48 × 48 grayscale
                                            ├─ predict with the CNN
                                            └─ draw label + confidences
```

## 🖥️ Alternative: native OpenCV webcam window

The inference module can also run as a standalone OpenCV program:

```bash
cd src
python Inference.py
```

Useful options:

```bash
# Find available webcam indices
python Inference.py --probe

# Use another webcam
python Inference.py --camera 1

# Use a different saved Keras model
python Inference.py --model ../Models/final_trained_model.keras
```

Press <kbd>Q</kbd> or <kbd>Esc</kbd> to close the native webcam window.

## 📦 Dataset

The model was designed using the **FER-2013 Facial Expression Recognition** dataset. Download it from Kaggle:

➡️ **[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)**

The dataset is not required for real-time inference because the trained model is included in `Models/`. It is needed only if you want to reproduce the cleaning and training workflow.

After downloading and extracting it, organise it as follows (rename Kaggle's `train` and `test` folders if necessary):

```text
Data/
├── Train data/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── Test data/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

`Data/` is intentionally excluded from Git, so please download it directly from Kaggle and follow the dataset's licence and terms.

> [!NOTE]
> The training notebook uses working-directory-relative paths named `Train data` and `Test data`. Before retraining, either run the notebook with `Data/` as the working directory or update those paths to `../Data/Train data` and `../Data/Test data`.

## 🧠 Model and training approach

| Component | Details |
| --- | --- |
| Input | 48 × 48 grayscale face image |
| Architecture | Four CNN blocks with batch normalisation, ReLU, dropout, and global average pooling |
| Output | Seven-class softmax classifier |
| Parameters | 1,241,895 |
| Optimiser / loss | Adam / categorical cross-entropy |
| Augmentation | Random horizontal flip, rotation, zoom, and translation |
| Imbalance strategy | Unweighted warm-start followed by damped class weighting |

The `Rescaling(1./255)` layer is embedded in the saved model. Therefore, `Inference.py` supplies raw pixel values and does **not** normalise the face a second time.

### Data-quality work

Before training, the notebook checks image hashes, removes duplicate files, flags conflicting labels, and checks for corrupt images. The accompanying report records 1,853 duplicate images removed, including 531 train/test leaks, leaving a cleaned split of 26,942 training and 7,125 test images.

## 📊 Results

The final damped-class-weighted model achieved a **weighted F1 score of 0.60** on the cleaned FER-2013 evaluation data. It improved `disgust` recall from 0.00 to 0.18 while preserving the other emotion classes.

<p align="center">
  <img src="Outputs/Graphical%20analysis/Final%20plots.png" alt="Training and validation history" width="820" />
</p>

<p align="center">
  <img src="Outputs/Graphical%20analysis/Both_correlation_matrix.png" alt="Confusion matrices before and after class weighting" width="820" />
</p>

More detail is available in [the final project report](Report/Final%20report.pdf) and the [training notebook](Notebooks/Model_Training.ipynb).

## 🗂️ Repository structure

```text
.
├── Data/                         # FER-2013 images (ignored by Git)
│   ├── Train data/
│   └── Test data/
├── Models/
│   ├── final_trained_model.keras # Model used by the applications
│   └── Final_history_model_training.pkl
├── Notebooks/
│   └── Model_Training.ipynb       # Cleaning, training, evaluation, plots
├── Outputs/
│   ├── Graphical analysis/        # Training and confusion-matrix figures
│   └── saved model/history files
├── Report/
│   └── Final report.pdf
├── src/
│   ├── Inference.py               # Reusable inference utilities + native webcam mode
│   └── run.py                     # Streamlit real-time application
├── requirements.txt
└── README.md
```

## 🛠️ Requirements

- Python 3.10 or newer
- A webcam and permission to use it in your browser
- The packages listed in [`requirements.txt`](requirements.txt)
- `Models/final_trained_model.keras` present in the cloned repository

For model training, install the same requirements and download the FER-2013 dataset. Training is substantially more resource-intensive than running the pre-trained model.

## 🧯 Troubleshooting

| Issue | What to check |
| --- | --- |
| `Failed to load model` | Confirm that `Models/final_trained_model.keras` was committed and downloaded with the repository. |
| No camera video in Streamlit | Allow camera access in the browser, then reload the page. |
| Wrong camera selected | Run `python Inference.py --probe`, then use `--camera <index>` for the native OpenCV app. |
| Module not found | Activate the virtual environment and run `pip install -r requirements.txt`. |

## ⚠️ Limitations

- FER-2013 contains noisy and imbalanced labels; `disgust` remains the hardest class.
- Haar cascades can be less reliable with poor lighting, occlusion, and non-frontal faces.
- Predictions are demonstrations of model output, not a clinical, psychological, or biometric assessment.

## 🙌 Acknowledgements

- [FER-2013 dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- TensorFlow / Keras, OpenCV, Streamlit, and `streamlit-webrtc`

