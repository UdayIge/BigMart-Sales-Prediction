# 🛒 BigMart Sales Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-orange.svg)
![License](https://img.shields.io/badge/License-Apache-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

**A comprehensive machine learning project for predicting sales at BigMart outlets**

[Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) • [Documentation](README.md)

</div>

## 📋 Project Overview

This project demonstrates a complete **end-to-end data science workflow** for predicting sales at BigMart outlets. Using advanced machine learning techniques, we analyze historical sales data to build predictive models that can forecast item outlet sales with high accuracy.

### 🎯 Key Objectives
- **Data Analysis**: Comprehensive exploratory data analysis with 18+ visualizations
- **ML Pipeline**: Multiple machine learning algorithms with model comparison
- **Interactive Dashboard**: Real-time predictions and data exploration
- **Business Insights**: Actionable recommendations for sales optimization

## 📊 Dataset

<div align="center">

| **Metric** | **Value** |
|------------|-----------|
| **Training Samples** | 8,523 |
| **Test Samples** | 5,681 |
| **Features** | 12 → 47 (after engineering) |
| **Target Variable** | Item_Outlet_Sales |
| **Missing Values** | Handled (Item_Weight: 17%, Outlet_Size: 28%) |

</div>

### 🔑 Key Features
- **Item Characteristics**: Weight, fat content, visibility, type, MRP
- **Outlet Information**: Size, location type, establishment year, outlet type
- **Target Variable**: Item outlet sales (continuous)

### 📈 Data Quality
- **Sales Range**: $33.29 to $13,086.96
- **Average Sales**: $2,181.29
- **Data Types**: Mixed (numerical, categorical, temporal)
- **Preprocessing**: Missing value imputation, feature engineering, encoding

## 🏗️ Project Structure

```
📁 BigMart-Sales-Prediction/
├── data/                          # Data files
│   ├── Train.csv                     # Original training dataset
│   ├── Test.csv                      # Original test dataset
│   ├── processed_train.csv           # Preprocessed training data
│   └── processed_test.csv            # Preprocessed test data
├── src/                           # Source code modules
│   ├── data_preprocessing.py         # Data cleaning & feature engineering
│   ├── eda.py                        # Exploratory data analysis
│   ├── model_training.py             # Advanced ML models
│   └── simple_model_training.py      # Basic ML models
├── dashboard/                     # Interactive web application
│   └── app.py                        # Streamlit dashboard
├── models/                        # Trained ML models
│   ├── best_model.pkl                # Best performing model
│   └── *.pkl                         # All trained models
├── results/                       # Analysis results & visualizations
│   ├── *.png                         # Static visualizations
│   ├── *.html                        # Interactive plots
│   └── model_performance.csv         # Model comparison results
├── notebooks/                     # Jupyter notebooks
│   └── analysis.ipynb                # Interactive analysis notebook
├── requirements.txt               # Python dependencies
├── main.py                        # Complete pipeline runner
├── run_dashboard.py               # Dashboard launcher
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### 📋 Prerequisites

- **Python**: 3.8 or higher
- **Package Manager**: pip
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB free space

### ⚡ Installation

<details>
<summary><b>🔽 Click to expand installation steps</b></summary>

1. **Clone the repository**
   ```bash
   git clone https://github.com/UdayIge/BigMart-Sales-Prediction.git
   cd BigMart-Sales-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download from [Kaggle Competition](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
   - Place `Train.csv` and `Test.csv` in the `data/` directory

</details>

### 🎯 Running the Project

#### **Option 1: Complete Pipeline (Recommended)**
```bash
python main.py
```

#### **Option 2: Individual Components**
```bash
# 1. Data Preprocessing
python src/data_preprocessing.py

# 2. Exploratory Data Analysis  
python src/eda.py

# 3. Model Training
python src/simple_model_training.py

# 4. Launch Dashboard
python run_dashboard.py
```

#### **Option 3: Interactive Dashboard**
```bash
streamlit run dashboard/app.py
```
> 🌐 Dashboard will open at http://localhost:8501

<!-- ## 📈 Analysis Pipeline

<div align="center">

```mermaid
graph LR
    A[📊 Raw Data] -> B[🔧 Preprocessing]
    B -> C[🔍 EDA]
    C -> D[🤖 ML Models]
    D -> E[📊 Dashboard]
    E -> F[💼 Insights]
```

</div> -->

### 🔧 1. Data Preprocessing

<details>
<summary><b>📋 Click to see preprocessing details</b></summary>

**Features:**
- ✅ Missing value imputation (Item_Weight: 17%, Outlet_Size: 28%)
- ✅ Categorical variable cleaning and standardization
- ✅ Feature engineering (outlet age, price categories, visibility categories)
- ✅ One-hot encoding for categorical variables
- ✅ Data validation and quality checks

**Output:** 47 engineered features from 12 original features

</details>

### 🔍 2. Exploratory Data Analysis

<details>
<summary><b>📊 Click to see EDA details</b></summary>

**Generated Visualizations:**
- ✅ 18+ static charts (PNG files)
- ✅ 3 interactive plots (HTML files)
- ✅ Statistical summaries and distributions
- ✅ Correlation matrices and heatmaps
- ✅ Business pattern analysis

**Key Insights:**
- Sales range: $33.29 to $13,086.96
- Best item type: Starchy Foods ($2,374.33)
- Best outlet type: Supermarket Type3 ($3,694.04)
- Price-sales correlation: 0.568

</details>

### 🤖 3. Machine Learning Models

<div align="center">

| **Model** | **R² Score** | **RMSE** | **MAE** | **Status** |
|-----------|--------------|----------|---------|------------|
| **Gradient Boosting** | **0.603** | **1039.33** | **723.62** | 🏆 **Best** |
| Linear Regression | 0.578 | 1071.54 | 794.98 | ✅ |
| Ridge Regression | 0.578 | 1071.56 | 794.99 | ✅ |
| Random Forest | 0.559 | 1094.94 | 765.35 | ✅ |
| Decision Tree | 0.179 | 1493.97 | 1033.22 | ⚠️ |

</div>

<details>
<summary><b>🔽 Click to see all models</b></summary>

**Implemented Algorithms:**
- Linear Regression, Ridge, Lasso, Elastic Net
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, XGBoost, LightGBM
- Support Vector Regression, K-Nearest Neighbors

**Features:**
- Cross-validation (5-fold)
- Hyperparameter tuning
- Feature importance analysis
- Model comparison and evaluation

</details>

### 🎨 4. Interactive Dashboard

<details>
<summary><b>🖥️ Click to see dashboard features</b></summary>

**5 Interactive Pages:**
- 📊 **Data Overview**: Dataset summary and statistics
- 🔍 **Exploratory Analysis**: Interactive visualizations
- 🤖 **Model Performance**: Model comparison charts
- 📈 **Predictions**: Real-time prediction interface
- 📋 **Insights**: Key findings and recommendations

**Features:**
- Real-time sales predictions
- Interactive sliders and dropdowns
- Batch predictions for test data
- Download functionality for results
- Professional UI/UX design

</details>

## 🏆 Key Results

<div align="center">

### 📊 Model Performance

| **Metric** | **Best Value** | **Model** |
|------------|----------------|-----------|
| **R² Score** | **0.603** | Gradient Boosting |
| **RMSE** | **1039.33** | Gradient Boosting |
| **MAE** | **723.62** | Gradient Boosting |

</div>

### 💡 Business Insights

<div align="center">

| **Category** | **Top Performer** | **Value** |
|--------------|-------------------|-----------|
| **Item Type** | Starchy Foods | $2,374.33 |
| **Outlet Type** | Supermarket Type3 | $3,694.04 |
| **Price Correlation** | Strong Positive | 0.568 |

</div>

### 🎯 Key Findings

<details>
<summary><b>📈 Click to see detailed insights</b></summary>

**Data Insights:**
- Sales range: $33.29 to $13,086.96 (mean: $2,181.29)
- Missing values: Item_Weight (17%), Outlet_Size (28%)
- Feature engineering: 47 features from 12 original
- Data quality: High with proper preprocessing

**Model Performance:**
- Best model: Gradient Boosting (R² = 0.603)
- Feature importance: Item MRP, Outlet Type, Item Type
- Cross-validation: Consistent performance (CV = 0.593)
- Robust predictions with low RMSE (1039.33)

**Business Recommendations:**
1. **Focus on Starchy Foods** - Highest average sales
2. **Invest in Supermarket Type3** - Best performing outlet type
3. **Price Optimization** - Strong correlation with sales
4. **Outlet Age Monitoring** - Consider establishment year impact
5. **Visibility Strategy** - Target high-visibility items

</details>

## 🛠️ Tech Stack

<div align="center">

| **Category** | **Technologies** |
|--------------|------------------|
| **🔧 Core** | Python 3.8+, Pandas, NumPy |
| **🤖 ML** | Scikit-learn, XGBoost, LightGBM |
| **📊 Viz** | Matplotlib, Seaborn, Plotly |
| **🌐 Web** | Streamlit, HTML/CSS |
| **📓 Tools** | Jupyter, Git, VS Code |

</div>

### 📦 Key Dependencies

```python
# Core Data Science
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Machine Learning
xgboost>=1.7.0
lightgbm>=4.0.0

# Web Dashboard
streamlit>=1.28.0
```

## 📁 Project Outputs

<div align="center">

| **Category** | **Files** | **Description** |
|--------------|-----------|-----------------|
| **📊 Data** | 5 files | Processed datasets & feature names |
| **🤖 Models** | 9 files | Trained ML models & scalers |
| **📈 Results** | 20+ files | Visualizations & performance metrics |
| **📓 Docs** | 4 files | Documentation & notebooks |

</div>

### 📊 Generated Files

<details>
<summary><b>📋 Click to see all generated files</b></summary>

**Data Files:**
- `data/processed_train.csv` - Clean training data (8,523 × 47)
- `data/processed_test.csv` - Clean test data (5,681 × 46)
- `data/feature_names.csv` - Feature names list

**Model Files:**
- `models/best_model.pkl` - Best performing model (Gradient Boosting)
- `models/gradient_boosting.pkl` - Gradient Boosting model
- `models/random_forest.pkl` - Random Forest model
- `models/linear_regression.pkl` - Linear Regression model
- `models/*.pkl` - All other trained models

**Results Files:**
- `results/model_performance.csv` - Model comparison results
- `results/*.png` - 18 static visualizations
- `results/*.html` - 3 interactive plots
- `results/submission.csv` - Test predictions

**Documentation:**
- `README.md` - Complete project documentation
- `PROJECT_SUMMARY.md` - Detailed project overview
- `notebooks/analysis.ipynb` - Interactive analysis notebook

</details>

## 🎓 Academic & Professional Value

### 🏆 What This Project Demonstrates

<div align="center">

| **Skill Category** | **Technologies & Techniques** |
|-------------------|-------------------------------|
| **📊 Data Analysis** | EDA, Statistical Analysis, Data Visualization |
| **🤖 Machine Learning** | Multiple Algorithms, Model Selection, Evaluation |
| **💻 Software Development** | Modular Design, Error Handling, Documentation |
| **🌐 Web Development** | Streamlit, Interactive Dashboards, UI/UX |
| **📈 Business Intelligence** | Insights Generation, Recommendations |

</div>

### 🎯 Perfect For

- **🎓 Final Year Projects** - Comprehensive end-to-end demonstration
- **💼 Portfolio Projects** - Professional-grade implementation
- **📚 Learning** - Data science best practices and workflows
- **🏢 Industry Applications** - Real-world sales prediction scenarios

## 🚀 Live Demo

<div align="center">

### 🎨 Interactive Dashboard

**🌐 [Launch Dashboard](http://localhost:8501)**

*Features: Real-time predictions, interactive visualizations, model performance comparison*

</div>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

<div align="center">

**🎓 Final Year Data Science Project**

| **Role** | **Contact** |
|----------|-------------|
| **📧 Email** | [udayige1696@gmail.com](mailto:udayige1696@gmail.com) |
| **🐙 GitHub** | [@udayige](https://github.com/udayige) |
| **💼 LinkedIn** | [LinkedIn](https://linkedin.com/) |

</div>

## 🙏 Acknowledgments

- **🏆 Kaggle** - For providing the BigMart Sales Prediction dataset
- **🏢 BigMart** - For the real-world business context
- **🌍 Open Source Community** - For the amazing tools and libraries
- **📚 Contributors** - Everyone who helped improve this project

## 🔮 Future Enhancements

<details>
<summary><b>🚀 Click to see planned features</b></summary>

**Technical Improvements:**
- [ ] Real-time data integration
- [ ] Advanced feature engineering
- [ ] Deep learning models (Neural Networks)
- [ ] Automated model deployment
- [ ] API development
- [ ] Mobile app development

**Business Features:**
- [ ] Customer segmentation
- [ ] Demand forecasting
- [ ] Price optimization
- [ ] Inventory management
- [ ] Performance monitoring

**User Experience:**
- [ ] Advanced filtering options
- [ ] Custom report generation
- [ ] Email notifications
- [ ] Multi-language support

</details>

## 📊 Project Statistics

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/udayige/BigMart-Sales-Prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/udayige/BigMart-Sales-Prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/udayige/BigMart-Sales-Prediction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/udayige/BigMart-Sales-Prediction)

**⭐ If you found this project helpful, please give it a star!**

</div>

---

<div align="center">

**🎉 Thank you for exploring the BigMart Sales Prediction project!**

*Built with ❤️ for the data science community*

[⬆️ Back to Top](#-bigmart-sales-prediction)

</div>
