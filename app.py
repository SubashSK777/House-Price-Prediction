import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

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

# Header (Banner Removed)
st.markdown("<h1 style='text-align: center; color: #FF5722;'>🏠 House Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Explore data, clean features, and build advanced regressions.</p>", unsafe_allow_html=True)

# Navigation setup (Spacious Sidebar)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.title("🛠️ Project Steps")
steps = [
    "0. Environment Setup",
    "1. Loading & Initial Inspection",
    "2. Exploratory Data Analysis (EDA)",
    "3. Data Cleaning & Imputation",
    "4. Feature Engineering",
    "5. Training & Model Selection",
    "6. Visualizing Results"
]

if 'step_index' not in st.session_state:
    st.session_state.step_index = 0

def next_step():
    st.session_state.step_index = min(st.session_state.step_index + 1, len(steps) - 1)

step = st.sidebar.radio("Navigate Workflow:", steps, index=st.session_state.step_index)
st.session_state.step_index = steps.index(step)
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)

# Initialize Session State
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'df_test' not in st.session_state:
    st.session_state.df_test = None

# --- Content ---

if step == "0. Environment Setup":
    st.header("🛠️ Step 0: Environment Setup")
    st.write("Loading essential toolkits for data manipulation and visualization.")
    if st.button("✅ Run Setup"):
        warnings.filterwarnings('ignore')
        sns.set_theme(style="whitegrid")
        with st.expander("🔍 Setup Details", expanded=True):
            st.success("Libraries loaded successfully!")
            st.code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns")
    
    if st.button("➡️ Next Part", key="next_0"):
        next_step(); st.rerun()

elif step == "1. Loading & Initial Inspection":
    st.header("📂 Step 1: Loading & Initial Inspection")
    
    if st.button("🚀 Load Dataset"):
        try:
            df_train = pd.read_csv("train.csv")
            df_test = pd.read_csv("test.csv")
            st.session_state.df_train = df_train
            st.session_state.df_test = df_test
            with st.expander("📊 Inspection Result", expanded=True):
                st.write(f"🚀 **Success!** Train: {df_train.shape} | Test: {df_test.shape}")
                st.dataframe(df_train.head())
        except Exception as e:
            st.error(f"⚠️ Error: {e}. Please ensure 'train.csv' is present in the project folder.")

    if st.session_state.df_train is not None:
        if st.button("➡️ Next Part", key="next_1"):
            next_step(); st.rerun()

elif step == "2. Exploratory Data Analysis (EDA)":
    st.header("📊 Step 2: Exploratory Data Analysis (EDA)")
    if st.session_state.df_train is None:
        st.warning("Please load data in Step 1 first.")
    else:
        if st.button("📈 Analyze Distribution"):
            with st.expander("🎨 Distribution Plots", expanded=True):
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                sns.histplot(st.session_state.df_train['SalePrice'], kde=True, color='teal', ax=ax[0])
                ax[0].set_title("Original")
                sns.histplot(np.log1p(st.session_state.df_train['SalePrice']), kde=True, color='orange', ax=ax[1])
                ax[1].set_title("Log-Transformed")
                st.pyplot(fig)

        if st.button("🔥 Correlations"):
            with st.expander("🌡️ Correlation Heatmap", expanded=True):
                num_train = st.session_state.df_train.select_dtypes(include=[np.number])
                corr = num_train.corr()['SalePrice'].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 10))
                sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
        
        if st.button("➡️ Next Part", key="next_2"):
            next_step(); st.rerun()

elif step == "3. Data Cleaning & Imputation":
    st.header("🧼 Step 3: Data Cleaning & Imputation")
    if st.session_state.df_train is None:
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
            with st.expander("✨ Cleaning Details", expanded=True):
                st.success("Missing values handled!")
                st.write(all_data.isnull().sum().sort_values(ascending=False).head(5))

        if 'all_data' in st.session_state:
            if st.button("➡️ Next Part", key="next_3"):
                next_step(); st.rerun()

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
            with st.expander("🏗️ Engineering Result", expanded=False):
                st.success("Sophisticated features engineered!")
                st.dataframe(all_data[['TotalSF', 'HouseAge', 'TotalBaths']].head())
        
        if st.button("➡️ Next Part", key="next_4"):
            next_step()
            st.rerun()

elif step == "5. Training & Model Selection":
    st.header("🤖 Step 5: Training & Model Selection")
    if 'all_data' not in st.session_state:
        st.warning("Please engineer features in Step 4 first.")
    else:
        if st.button("🧠 Train Models"):
            with st.expander("🤖 Training Result", expanded=False):
                all_data = st.session_state.all_data
                df_train = st.session_state.df_train
                features = ['TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBaths', 'HouseAge', 'YearsRemod', 'TotalBsmtSF', 'HasPool', 'HasGarage']
                X = all_data[:len(df_train)][features]
                y = np.log1p(df_train['SalePrice'])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                models = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=10), "Lasso Regression": Lasso(alpha=0.001)}
                results = []
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    r2 = r2_score(y_test, preds)
                    results.append({"Model": name, "RMSE": round(rmse, 4), "R2": round(r2, 4)})
                
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test
                st.session_state.features = features
                st.table(pd.DataFrame(results))
        
        if 'X_train' in st.session_state:
            if st.button("➡️ Next Part", key="next_5"):
                next_step()
                st.rerun()

elif step == "6. Visualizing Results":
    st.header("📈 Step 6: Visualizing Results")
    if 'X_train' not in st.session_state:
        st.warning("Please train models in Step 5 first.")
    else:
        if st.button("🎯 Regression Performance"):
            with st.expander("📈 Comparison Plot", expanded=False):
                best_model = Ridge(alpha=10)
                best_model.fit(st.session_state.X_train, st.session_state.y_train)
                final_preds = best_model.predict(st.session_state.X_test)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x=st.session_state.y_test, y=final_preds, line_kws={"color": "red", "linestyle": "--"}, scatter_kws={"alpha": 0.5, "color": "teal"}, ax=ax)
                st.pyplot(fig)

        if st.button("🏆 Feature Importance"):
            with st.expander("🥇 Importance Chart", expanded=False):
                best_model = Ridge(alpha=10)
                best_model.fit(st.session_state.X_train, st.session_state.y_train)
                coef_df = pd.DataFrame({'Feature': st.session_state.features, 'Importance': best_model.coef_}).sort_values('Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=coef_df, x='Importance', y='Feature', palette='magma', ax=ax)
                st.pyplot(fig)
        
        st.success("🎉 Final Step reached!")
        if st.button("🚀 Restart Process"):
            st.session_state.step_index = 0
            st.rerun()
