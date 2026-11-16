# Handwritten Digit Classification

A beginner-friendly B.Tech Machine Learning project that predicts handwritten digits (0–9) using SVM and Random Forest, with a Streamlit dashboard.

## Dataset
Uses `sklearn.datasets.load_digits`, a built-in dataset of 1,797 grayscale digit images. Each image is 8×8 pixels, so no separate dataset download is required.

## Features
- Upload an image containing one handwritten digit.
- Predict using SVM or Random Forest.
- View prediction probabilities.
- Compare real held-out test accuracy.
- Retrain both models with one command.

## Windows setup (VS Code / PowerShell)

1. Extract the ZIP file.
2. Open the extracted `Handwritten-Digit-Classification` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

5. Start the dashboard:

```powershell
python -m streamlit run app.py
```

6. Open the local address printed in the terminal (usually http://localhost:8501).

## Retrain the models
With `.venv` activated:

```powershell
python train_model.py
```

Then restart Streamlit if it is already running.

## Important note
This project uses Scikit-learn's 8×8 Digits dataset, not the 28×28 MNIST dataset. For best upload results, use a single large, centered digit with strong contrast. This is an educational project and not intended for medical, financial, or other high-stakes decisions.
