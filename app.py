import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #050505;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF5722;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e64a19;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div align="center">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=House%20Price%20Prediction&fontSize=50&animation=fadeIn" width="100%" />
    <p align="center">
      <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&pause=1000&color=FF5722&center=true&vCenter=true&width=600&lines=Analyze+Predictive+Patterns;Build+Advanced+Regressions;Predict+Prices+with+Precision!" alt="Typing SVG" />
    </p>
    </div>
    """, unsafe_allow_html=True)

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Sidebar for Navigation / Steps
st.sidebar.title("🛠️ Project Steps")
step = st.sidebar.radio("Go to Step:", [
    "0. Environment Setup",
    "1. Loading & Initial Inspection",
    "2. Exploratory Data Analysis (EDA)",
    "3. Data Cleaning & Imputation",
    "4. Feature Engineering",
    "5. Training & Model Selection",
    "6. Visualizing Results"
])

# --- Content ---

if step == "0. Environment Setup":
    st.header("🛠️ Step 0: Environment Setup")
    st.write("Loading essential toolkits for data manipulation and visualization.")
    if st.button("✅ Run Setup"):
        warnings.filterwarnings('ignore')
        sns.set_theme(style="whitegrid")
        st.success("Libraries loaded successfully!")
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
        """)

elif step == "1. Loading & Initial Inspection":
    st.header("📂 Step 1: Loading & Initial Inspection")
    if st.button("🚀 Load Dataset"):
        try:
            df_train = pd.read_csv("train.csv")
            df_test = pd.read_csv("test.csv")
            st.session_state.df_train = df_train
            st.session_state.df_test = df_test
            st.write(f"**Train shape:** {df_train.shape}")
            st.write(f"**Test shape:** {df_test.shape}")
            st.dataframe(df_train.head())
        except Exception as e:
            st.error(f"Error loading data: {e}. Please ensure train.csv and test.csv are in the project folder.")

elif step == "2. Exploratory Data Analysis (EDA)":
    st.header("📊 Step 2: Exploratory Data Analysis (EDA)")
    if 'df_train' not in st.session_state:
        st.warning("Please load data in Step 1 first.")
    else:
        if st.button("📈 Analyze SalePrice Distribution"):
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(st.session_state.df_train['SalePrice'], kde=True, color='teal', ax=ax[0])
            ax[0].set_title("Original SalePrice Distribution")
            sns.histplot(np.log1p(st.session_state.df_train['SalePrice']), kde=True, color='orange', ax=ax[1])
            ax[1].set_title("Log-Transformed SalePrice Distribution")
            st.pyplot(fig)

        if st.button("🔥 Show Feature Correlations"):
            num_train = st.session_state.df_train.select_dtypes(include=[np.number])
            corr = num_train.corr()['SalePrice'].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 10))
            sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

elif step == "3. Data Cleaning & Imputation":
    st.header("🧼 Step 3: Data Cleaning & Imputation")
    if 'df_train' not in st.session_state:
        st.warning("Please load data in Step 1 first.")
    else:
        if st.button("✨ Clean Data"):
            df_train = st.session_state.df_train
            df_test = st.session_state.df_test
            all_data = pd.concat([df_train.drop('SalePrice', axis=1), df_test], axis=0).reset_index(drop=True)
            
            # Cleaning Logic
            none_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']
            for col in none_cols: all_data[col] = all_data[col].fillna("None")
            zero_cols = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
            for col in zero_cols: all_data[col] = all_data[col].fillna(0)
            all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
            for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
                all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
            
            st.session_state.all_data = all_data
            st.success("Missing values handled!")
            st.write(all_data.isnull().sum().sort_values(ascending=False).head(10))

elif step == "4. Feature Engineering":
    st.header("🏗️ Step 4: Feature Engineering")
    if 'all_data' not in st.session_state:
        st.warning("Please clean data in Step 3 first.")
    else:
        if st.button("🚀 Engineer Features"):
            all_data = st.session_state.all_data
            all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
            all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
            all_data['YearsRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']
            all_data['TotalBaths'] = all_data['FullBath'] + all_data['BsmtFullBath'] + 0.5 * (all_data['HalfBath'] + all_data['BsmtHalfBath'])
            all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
            all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
            st.session_state.all_data = all_data
            st.success("Sophisticated features engineered!")
            st.dataframe(all_data[['TotalSF', 'HouseAge', 'TotalBaths']].head())

elif step == "5. Training & Model Selection":
    st.header("🤖 Step 5: Training & Model Selection")
    if 'all_data' not in st.session_state:
        st.warning("Please engineer features in Step 4 first.")
    else:
        if st.button("🧠 Train Models"):
            all_data = st.session_state.all_data
            df_train = st.session_state.df_train
            features = ['TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBaths', 'HouseAge', 'YearsRemod', 'TotalBsmtSF', 'HasPool', 'HasGarage']
            X = all_data[:len(df_train)][features]
            y = np.log1p(df_train['SalePrice'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=10),
                "Lasso Regression": Lasso(alpha=0.001)
            }
            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                results.append({"Model": name, "RMSE": round(rmse, 4), "R2": round(r2, 4)})
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.features = features
            st.table(pd.DataFrame(results))

elif step == "6. Visualizing Results":
    st.header("📈 Step 6: Visualizing Results")
    if 'X_train' not in st.session_state:
        st.warning("Please train models in Step 5 first.")
    else:
        if st.button("🎯 Show Regression Performance"):
            best_model = Ridge(alpha=10)
            best_model.fit(st.session_state.X_train, st.session_state.y_train)
            final_preds = best_model.predict(st.session_state.X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=st.session_state.y_test, y=final_preds, line_kws={"color": "red", "linestyle": "--"}, scatter_kws={"alpha": 0.5, "color": "teal"}, ax=ax)
            ax.set_xlabel("Actual Log Price")
            ax.set_ylabel("Predicted Log Price")
            ax.set_title("Actual vs Predicted House Prices")
            st.pyplot(fig)

        if st.button("🏆 Show Feature Importance"):
            best_model = Ridge(alpha=10)
            best_model.fit(st.session_state.X_train, st.session_state.y_train)
            coef_df = pd.DataFrame({'Feature': st.session_state.features, 'Importance': best_model.coef_}).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coef_df, x='Importance', y='Feature', palette='magma', ax=ax)
            st.pyplot(fig)
