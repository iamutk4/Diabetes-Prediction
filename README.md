# Diabetes Prediction using Support Vector Machine (SVM)

This project aims to predict whether a patient has diabetes or not based on various health-related features. The dataset consists of 768 instances with 8 features including pregnancy, glucose levels, blood pressure, skin thickness, insulin levels, BMI (Body Mass Index), Diabetes Pedigree Function, and age.

## Workflow Overview:

1. **Exploratory Data Analysis (EDA):** Conducted to gain insights into the dataset's characteristics, distributions, and correlations among features. This step helps in understanding the data better and making informed decisions during model building.

2. **Data Preprocessing:** Standardization is performed to bring all features to the same scale, which is essential for many machine learning algorithms, including Support Vector Machines (SVM).

3. **Train-Test Split:** The dataset is divided into training and testing sets with a test size of 20%. This ensures that the model's performance is evaluated on unseen data, helping to assess its generalization ability.

4. **Model Training:** Utilizing the Support Vector Machine (SVM) algorithm, the model is trained on the training data. SVM is chosen for its ability to handle both linear and non-linear classification tasks effectively.

5. **Model Evaluation:** The accuracy of the trained model is evaluated on both the training and testing datasets. Accuracy score serves as a metric to measure the model's performance in predicting diabetes.

![screenshot](https://github.com/iamutk4/Diabetes-Prediction/assets/69798933/fc5aceb3-1171-4521-8805-b458d65e5068)


## Libraries Used:

- `numpy`: For numerical computations
- `pandas`: For data manipulation and analysis
- `sklearn`: For machine learning tasks
  - `StandardScaler`: For feature scaling
  - `train_test_split`: For splitting the dataset
  - `svm`: For implementing Support Vector Machine algorithm
  - `accuracy_score`: For evaluating the model's accuracy

## Dataset:

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). It consists of 768 instances with 8 features and 1 target variable (Outcome - whether the patient has diabetes or not).



## Usage:

1. Clone the repository:

```
git clone https://github.com/iamutk4/Diabetes-Prediction.git
```
2. Navigate to project directory
   ```
   cd diabetes-prediction
   ```
3. Open the Jupyter notebook Diabetes_Prediction.ipynb using Jupyter Notebook or JupyterLab.
