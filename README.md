# FaceSense AI — Facial Expression Recognition

A portfolio-ready learning project using a CNN trained on FER-2013, OpenCV face detection,
a Streamlit dashboard, and a separate real-time webcam application.

## Features
- Seven expression classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- CNN training and evaluation
- Confusion matrix, classification report, training history
- Webcam expression demo
- Streamlit image upload and camera snapshot interface

> Important: the model classifies visible facial-expression patterns. It does not establish
> what a person truly feels, intends, or experiences.

## 1. Requirements
- Python 3.10 or 3.11 recommended for this pinned TensorFlow setup
- VS Code
- Webcam (for live demo)
- Internet connection for installing packages and downloading the dataset

## 2. Create environment (Windows PowerShell)
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

## 3. Dataset
Download the FER-2013 dataset CSV from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

Place the downloaded CSV at:
`data/fer2013.csv`

The CSV must contain `emotion` and `pixels` columns. The standard FER-2013 CSV also
contains `Usage`, which this project uses for the provided train/test split.

Do not upload the dataset to GitHub unless its license and distribution terms permit it.

## 4. Train the CNN
From the project root:
```powershell
python src/train.py
```
Training can take a while. A GPU is optional; CPU training is slower. The best model is saved to
`models/facial_emotion_model.keras`. Evaluation files are saved in `reports/`.

## 5. Run real-time webcam detection
```powershell
python webcam.py
```
Press `Q` to close the webcam window.

## 6. Run the web dashboard
```powershell
streamlit run app.py
```
Open the local URL printed in the terminal. The dashboard supports image upload and camera
snapshots. The separate `webcam.py` script provides continuous real-time webcam detection.

## 7. GitHub
Before pushing, confirm that the dataset and model files are ignored by `.gitignore`.
```powershell
git init
git add .
git commit -m "Build FaceSense AI facial expression recognition"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Facial-Emotion-Recognition.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username and create the empty repository first.

## 8. Deployment notes
Streamlit Community Cloud can host the dashboard, but the trained model file must be made
available to the deployed app. Large model files may require Git LFS or a separate model
hosting/storage option. Webcam access on hosted websites depends on browser permissions and
is not equivalent to direct local OpenCV webcam access.

## Portfolio checklist
- Add screenshots of the dashboard and webcam demo
- Add measured test accuracy and classification report (never invent metrics)
- Explain dataset limitations and class imbalance
- Include a confusion matrix
- Record a short demo video
- Document known limitations and ethical use
