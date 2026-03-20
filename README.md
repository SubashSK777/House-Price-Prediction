# Project 01 · Beginner Friendly
## House Price Prediction

Predict house prices using regression — the most classic ML interview problem.

### Skills you'll learn
- Pandas & EDA
- Feature Engineering
- Linear Regression
- Model Evaluation
- Matplotlib / Seaborn

### Build steps
1. **Get data (Kaggle)**
2. **EDA & clean**
3. **Feature eng.**
4. **Train model**
5. **Evaluate & tune**

---

Perfect setup! Let's build this properly. Here's your complete step-by-step guide: Click through each step — every step has the exact code to paste into your notebook, plus tips for what to say in interviews!

---

**How to work through this:**

Follow the steps in order. Each numbered cell goes into a new cell in your `house_price.ipynb` notebook. Run each cell with **Shift + Enter** before moving to the next one.

When you get an error, paste it here and I'll fix it immediately. When you finish a step, come back and I'll help you understand what you just built so you can explain it in interviews. Let's go — start with Step 1! 🚀

### Set up your VS Code environment
Install the Jupyter extension and all required libraries so VS Code can run your notebook.

1. **Install Jupyter extension in VS Code**
   - Open VS Code → click the Extensions icon on the left sidebar (4 squares icon)
   - Search `Jupyter` → install the one by Microsoft
   - Search `Python` → install the one by Microsoft (if not already installed)

2. **Open terminal in VS Code**
   - Press `Ctrl + \`` (backtick key, top-left of keyboard) to open terminal directly in VS Code.

3. **Install all required libraries**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Create your project folder**
   ```bash
   mkdir house-price-project
   cd house-price-project
   ```
   All your files will live inside this folder. Keep it organized.

### Download the dataset from Kaggle
We'll use the famous Kaggle 'House Prices' dataset. It's free and used worldwide for learning.

1. **Go to Kaggle**
   - Visit [kaggle.com](https://www.kaggle.com) → create a free account if you don't have one
   - Search: `House Prices Advanced Regression Techniques`
   - Click the competition → go to the **Data** tab
   - Download `train.csv` and `test.csv`

2. **Place files in your project folder**
   - Move both CSV files into your `house-price-project` folder

3. **Create your notebook**
   - In VS Code: File → New File → save as `house_price.ipynb`
   - The `.ipynb` extension tells VS Code it's a Jupyter Notebook. You'll run code cell by cell — perfect for data science.

**About the dataset**
- 1,460 houses with 79 features (rooms, area, quality, year built, etc.)
- Target column: `SalePrice` — what we want to predict
- Mix of numerical and categorical columns — great for learning real EDA

### Load data & Exploratory Data Analysis
Understand your data before touching any model. This is the most important step — and the most asked-about in interviews.

**Cell 1 — Import libraries**
```python
# Run this in your first notebook cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
print("Libraries loaded!")
```

**Cell 2 — Load and inspect data**
```python
df = pd.read_csv("train.csv")

print("Shape:", df.shape)          # rows x columns
print(df.head())                   # first 5 rows
print(df.dtypes)                   # data types
print(df.describe())               # stats summary
```

**Cell 3 — Check missing values**
```python
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Visualize missing values
plt.figure(figsize=(12,5))
missing.plot(kind='bar')
plt.title("Missing Values per Column")
plt.show()
```

**Cell 4 — Explore target variable**
```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice Distribution")

plt.subplot(1,2,2)
sns.histplot(np.log1p(df['SalePrice']), kde=True)
plt.title("Log(SalePrice) Distribution")
plt.tight_layout()
plt.show()
```
Log transform makes skewed price data more normal — which helps linear models. You'll explain this in interviews!

### Handle missing values & clean data
Real datasets are messy. This step teaches you to fix that — a very common interview discussion point.

**Cell 5 — Fill missing values**
```python
# Columns where NaN means "None/No feature" (e.g. no pool, no garage)
none_cols = ['PoolQC','MiscFeature','Alley','Fence',
             'FireplaceQu','GarageType','GarageFinish',
             'GarageQual','GarageCond','BsmtQual','BsmtCond',
             'BsmtExposure','BsmtFinType1','BsmtFinType2',
             'MasVnrType']

for col in none_cols:
    df[col] = df[col].fillna("None")

# Numerical columns — fill with 0 (no garage = 0 cars)
zero_cols = ['GarageYrBlt','GarageArea','GarageCars',
             'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
             'TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
             'MasVnrArea']

for col in zero_cols:
    df[col] = df[col].fillna(0)

# Fill with median (safer than mean for skewed data)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Fill with mode (most common value) for categoricals
df['MSZoning']   = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print("Missing after cleaning:", df.isnull().sum().sum())
```
Always ask yourself: "WHY is this value missing?" That reasoning is what interviewers want to hear — not just that you used `fillna()`.

### Feature Engineering
Create new useful features from existing columns. This is where creativity meets data science.

**Cell 6 — Create new features**
```python
# Total square footage — better than 3 separate columns
df['TotalSF'] = (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])

# House age when sold
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# Years since last remodel
df['YearsRemod'] = df['YrSold'] - df['YearRemodAdd']

# Total bathrooms (full counts more than half)
df['TotalBaths'] = (df['FullBath'] + df['BsmtFullBath'] +
                    0.5 * (df['HalfBath'] + df['BsmtHalfBath']))

# Does it have a pool? (binary)
df['HasPool']   = (df['PoolArea'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)

print("New features added!")
```

**Cell 7 — Encode categorical columns**
```python
# Select only numeric columns for now (simple approach)
num_df = df.select_dtypes(include=[np.number]).copy()

# Check correlation with SalePrice
corr = num_df.corr()['SalePrice'].sort_values(ascending=False)
print(corr.head(15))
```

**Cell 8 — Correlation heatmap**
```python
top_feats = corr[1:11].index  # top 10 correlated features
plt.figure(figsize=(10,8))
sns.heatmap(num_df[top_feats].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```
Correlation heatmap is a classic interview visual. Be ready to explain what multicollinearity means and why it's a problem for linear regression.

### Train your ML model
Now we finally train! We'll build 3 models and compare them — exactly what interviewers want to see.

**Cell 9 — Prepare data for modelling**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Use numeric features only (after engineering)
features = ['TotalSF','OverallQual','GrLivArea','GarageCars',
            'TotalBaths','HouseAge','YearsRemod','TotalBsmtSF',
            'HasPool','HasGarage']

X = num_df[features].copy()
y = np.log1p(df['SalePrice'])   # log-transform target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

**Cell 10 — Train & compare 3 models**
```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)"       : Ridge(alpha=10),
    "Lasso (L1)"       : Lasso(alpha=0.001)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    results[name] = {"RMSE": round(rmse, 4), "R²": round(r2, 4)}
    print(f"{name:25s} → RMSE: {rmse:.4f}  R²: {r2:.4f}")
```
Expect R² around 0.85–0.90. That means your model explains ~87% of the variation in house prices — strong for a fresher project!

### Evaluate & visualize results
Always visualize your model's performance — interviewers love when you go beyond just printing numbers.

**Cell 11 — Actual vs Predicted plot**
```python
best_model = Ridge(alpha=10)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, preds, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual log(SalePrice)")
plt.ylabel("Predicted log(SalePrice)")
plt.title("Actual vs Predicted House Prices")
plt.show()
# Points near the red line = good predictions!
```

**Cell 12 — Residuals plot**
```python
residuals = y_test - preds

plt.figure(figsize=(8,4))
plt.scatter(preds, residuals, alpha=0.5, color='steelblue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
# Random scatter around 0 = model is working well
```

**Cell 13 — Feature importance**
```python
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': best_model.coef_
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='Blues_d')
plt.title("Feature Importance (Ridge Coefficients)")
plt.show()
```
This chart answers: "Which features matter most for predicting price?" — a direct interview question. `OverallQual` and `TotalSF` will likely top the list.

### Push to GitHub & write your README
A project without GitHub doesn't exist for recruiters. This step makes your work visible.

1. **Create a GitHub repo**
   - Go to [github.com](https://www.github.com) → click **New Repository**
   - Name it: `house-price-prediction`
   - Set to **Public** → **Create repository**

2. **Push your project from VS Code terminal**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: House price prediction project"
   git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git
   git push -u origin main
   ```

3. **Create a README.md file**
# House Price Prediction

## Overview
Predicted house sale prices using regression models on the Kaggle House Prices dataset (1,460 samples, 79 features).

## Results
| Model             | RMSE   | R²    |
|-------------------|--------|-------|
| Linear Regression | 0.XXXX | 0.XXX |
| Ridge Regression  | 0.XXXX | 0.XXX |
| Lasso Regression  | 0.XXXX | 0.XXX |

## Key Steps
- EDA & missing value analysis
- Feature engineering (TotalSF, HouseAge, TotalBaths)
- Log-transformed target variable
- Compared 3 regression models

## Tools
Python · Pandas · Scikit-learn · Matplotlib · Seaborn

*(Fill in your actual RMSE and R² numbers. A README with a results table shows professionalism — most freshers skip this entirely.)*

**Note:** Don't upload the Kaggle CSV files to GitHub — they're too large. Add a note in README saying "Download train.csv from Kaggle link here."
