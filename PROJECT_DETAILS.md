# Project details for college demonstration

## Aim
To develop a machine-learning application that recognizes handwritten numerical digits from 0 to 9.

## Algorithms
1. Support Vector Machine (SVM): learns a decision boundary that separates digit classes in feature space.
2. Random Forest: combines predictions from multiple decision trees to classify an input digit.

## Methodology
1. Load the built-in Scikit-learn Digits dataset.
2. Split data into training and testing sets using stratification.
3. Train SVM and Random Forest classifiers.
4. Evaluate each model using held-out test accuracy.
5. Save the trained models using Joblib.
6. Use Streamlit to upload an image and display the predicted digit.

## Viva points
- **What is classification?** A supervised learning task that assigns an input to one of predefined classes.
- **Why use a train-test split?** To evaluate performance on data not used during training.
- **What is SVM?** A supervised algorithm that finds a separating boundary with a maximum margin.
- **What is Random Forest?** An ensemble of decision trees whose outputs are combined for prediction.
- **Why scale features for SVM?** Scaling helps features contribute on comparable numerical ranges.
- **What is accuracy?** Correct predictions divided by total predictions, multiplied by 100.
- **Why save a model?** So the trained model can be reused without retraining every time.
- **What is a limitation?** The supplied dataset uses 8×8 images; real camera images can differ in scale, orientation, and background.
