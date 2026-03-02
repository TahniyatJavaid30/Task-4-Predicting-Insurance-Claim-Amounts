# Task-4-Predicting-Insurance-Claim-Amounts
Estimate the medical insurance claim amount based on personal data.

# 🏥 Insurance Claim Amount Prediction – Medical Cost Analysis

## 📌 Objective

The objective of this project is to:

* Predict **medical insurance charges** based on personal attributes.
* Train a **Linear Regression** model.
* Analyze how **BMI, Age, and Smoking Status** impact insurance costs.
* Evaluate model performance using **MAE** and **RMSE**.

This is a **Supervised Regression** problem.

---

## 📊 Dataset

Dataset used: **Medical Cost Personal Dataset**

The dataset contains the following features:

* Age
* Sex
* BMI (Body Mass Index)
* Children
* Smoker (Yes/No)
* Region
* Charges (Target Variable – Insurance Cost)

Target Variable:

* `charges` → Medical Insurance Claim Amount (Continuous Value)

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```id="w3m9az"
insurance-claim-prediction/
│
├── insurance_model.ipynb
├── insurance.csv
├── requirements.txt
└── README.md
```

---

# 🔎 Project Workflow

---

## 1️⃣ Data Loading

```python id="g4xvti"
import pandas as pd

df = pd.read_csv("insurance.csv")
df.head()
```

---

## 2️⃣ Data Cleaning & Preparation

* Checked for missing values
* Encoded categorical variables (`sex`, `smoker`, `region`)
* Converted categorical features using One-Hot Encoding

```python
df.isnull().sum()

df = pd.get_dummies(df, drop_first=True)
```

---

# 📊 Exploratory Data Analysis (EDA)

---

## 📍 Age vs Insurance Charges

![Image](https://www.researchgate.net/publication/360187490/figure/fig1/AS%3A11431281206301910%401700644739313/Scatter-plot-of-claim-size-vs-age-Figure-4-and-claim-size-vs-BMI-Figure-5.png)

![Image](https://www.researchgate.net/publication/349931966/figure/fig1/AS%3A999613989408769%401615337955863/Share-of-health-care-costs-in-different-age-groups-As-the-number-of-patients-in-each-age.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2Anx0wKiYUWlOqJ7_1uOPLRw.png)

![Image](https://python-charts.com/en/correlation/scatter-plot-regression-line-seaborn_files/figure-html/scatter-plot-seaborn-regression-line-group-palette.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="age", y="charges", data=df)
plt.title("Age vs Insurance Charges")
plt.show()
```

📌 Insight: Insurance charges generally increase with age.

---

## 📍 BMI vs Insurance Charges

![Image](https://www.researchgate.net/publication/379486812/figure/fig4/AS%3A11431281233469498%401712061175611/Scatter-plot-of-BMI-vs-Charges.jpg)

![Image](https://www.researchgate.net/publication/319388810/figure/fig1/AS%3A11431281210541721%401702056402900/Relationship-between-BMI-and-direct-health-care-costs-Footnote-The-solid-curve.tif)

![Image](https://www.researchgate.net/publication/383522368/figure/fig4/AS%3A11431281274483693%401724951121112/Scatter-Plot-with-Average-Line-Relationship-between-BMI-and-Health-Insurance-Charges.png)

![Image](https://k3-production-bucket.s3.us-east-1.amazonaws.com/uploads/nyZ6HJrBkMdSCvzvi_plot.png)

```python
sns.scatterplot(x="bmi", y="charges", data=df)
plt.title("BMI vs Insurance Charges")
plt.show()
```

📌 Insight: Higher BMI tends to increase medical costs.

---

## 📍 Smoking Status Impact

![Image](https://www.researchgate.net/publication/379486812/figure/fig3/AS%3A11431281233476966%401712061175375/Bar-Plot-of-Average-Premium-Prices-of-a-Smoker-and-Non-smoker.ppm)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1208/1%2AgNvUUp3VNCMDDeJRn1NWcw.png)

![Image](https://scx2.b-cdn.net/gfx/news/hires/2015/smokerstheob.png)

![Image](https://bmjopen.bmj.com/content/bmjopen/2/6/e001678/F2.large.jpg)

```python
sns.boxplot(x="smoker_yes", y="charges", data=df)
plt.title("Smoking Status vs Charges")
plt.show()
```

📌 Insight: Smokers have significantly higher insurance charges.

---

# 🤖 Model Training – Linear Regression

We trained a **Linear Regression** model to predict insurance charges.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

## ✅ Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
```

MAE measures the average absolute difference between predicted and actual values.

---

## ✅ Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
```

RMSE penalizes larger errors more than MAE.

---

# 📊 Feature Correlation Analysis

```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```

📌 Strongest predictors:

* Smoking Status
* Age
* BMI

---

# 🎯 Skills Demonstrated

✔ Regression Modeling
✔ Linear Regression Implementation
✔ Feature Correlation Analysis
✔ Scatter Plot & Box Plot Visualization
✔ Error Metrics (MAE & RMSE)
✔ Model Evaluation

---

# 🚀 Future Improvements

* Apply Polynomial Regression
* Try Random Forest Regressor
* Feature Scaling
* Hyperparameter Tuning
* Deploy as API using Flask / FastAPI

---

# 🏁 Conclusion

This project demonstrates a complete **end-to-end regression pipeline**, including:

* Data preprocessing
* Visualization of important features
* Linear Regression modeling
* Error evaluation using MAE and RMSE

It provides practical experience in **predictive modeling for healthcare cost estimation**, which is widely used in insurance and healthcare analytics.
