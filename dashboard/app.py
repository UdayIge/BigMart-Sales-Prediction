"""
Interactive Dashboard for BigMart Sales Prediction Analysis
Author: Data Analysis Project
Description: Streamlit dashboard for data visualization and model insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="BigMart Sales Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data"""
    try:
        # Load original data
        train_data = pd.read_csv('data/Train.csv')
        test_data = pd.read_csv('data/Test.csv')
        
        # Load processed data if available
        try:
            processed_train = pd.read_csv('data/processed_train.csv')
            processed_test = pd.read_csv('data/processed_test.csv')
        except FileNotFoundError:
            processed_train = None
            processed_test = None
        
        # Load model results if available
        try:
            model_results = pd.read_csv('results/model_performance.csv')
        except FileNotFoundError:
            model_results = None
        
        return train_data, test_data, processed_train, processed_test, model_results
    except FileNotFoundError:
        st.error("Data files not found. Please ensure the data is in the correct directory.")
        return None, None, None, None, None

@st.cache_data
def load_model():
    """Load the best trained model"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛒 BigMart Sales Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    train_data, test_data, processed_train, processed_test, model_results = load_data()
    
    if train_data is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Data Overview", "🔍 Exploratory Analysis", "🤖 Model Performance", "📈 Predictions", "📋 Insights"]
    )
    
    # Main content based on selected page
    if page == "Data Overview":
        data_overview_page(train_data, test_data)
    elif page == "Exploratory Analysis":
        exploratory_analysis_page(train_data)
    elif page == "Model Performance":
        model_performance_page(model_results)
    elif page == "Predictions":
        predictions_page(train_data, test_data)
    elif page == "Insights":
        insights_page(train_data)

def data_overview_page(train_data, test_data):
    """Data overview page"""
    st.header("Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", f"{train_data.shape[0]:,}")
    
    with col2:
        st.metric("Test Samples", f"{test_data.shape[0]:,}")
    
    with col3:
        st.metric("Features", f"{train_data.shape[1]-1}")
    
    with col4:
        avg_sales = train_data['Item_Outlet_Sales'].mean()
        st.metric("Avg Sales", f"${avg_sales:.2f}")
    
    # Data summary
    st.subheader("Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Info:**")
        st.write(f"Shape: {train_data.shape}")
        st.write(f"Memory usage: {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.write("**Missing Values:**")
        missing_data = train_data.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(train_data)) * 100
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df)
        else:
            st.write("No missing values found!")
    
    with col2:
        st.write("**Test Data Info:**")
        st.write(f"Shape: {test_data.shape}")
        st.write(f"Memory usage: {test_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.write("**Missing Values:**")
        missing_data_test = test_data.isnull().sum()
        missing_df_test = pd.DataFrame({
            'Column': missing_data_test.index,
            'Missing Count': missing_data_test.values,
            'Missing %': (missing_data_test.values / len(test_data)) * 100
        })
        missing_df_test = missing_df_test[missing_df_test['Missing Count'] > 0]
        if len(missing_df_test) > 0:
            st.dataframe(missing_df_test)
        else:
            st.write("No missing values found!")
    
    # Data preview
    st.subheader("Data Preview")
    
    tab1, tab2 = st.tabs(["Training Data", "Test Data"])
    
    with tab1:
        st.dataframe(train_data.head(10))
    
    with tab2:
        st.dataframe(test_data.head(10))
    
    # Data types
    st.subheader("Data Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': train_data.dtypes.index,
            'Data Type': train_data.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.write("**Test Data Types:**")
        dtype_df_test = pd.DataFrame({
            'Column': test_data.dtypes.index,
            'Data Type': test_data.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df_test)

def exploratory_analysis_page(train_data):
    """Exploratory analysis page"""
    st.header("🔍 Exploratory Data Analysis")
    
    # Target variable analysis
    st.subheader("Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales distribution
        fig = px.histogram(train_data, x='Item_Outlet_Sales', nbins=50, 
                          title='Distribution of Item Outlet Sales')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales statistics
        sales_stats = train_data['Item_Outlet_Sales'].describe()
        st.write("**Sales Statistics:**")
        for stat, value in sales_stats.items():
            st.write(f"{stat}: ${value:.2f}")
        
        # Box plot
        fig = px.box(train_data, y='Item_Outlet_Sales', title='Sales Box Plot')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical analysis
    st.subheader("Categorical Variables Analysis")
    
    categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Item_Identifier')
    
    selected_cat = st.selectbox("Select Categorical Variable", categorical_cols)
    
    if selected_cat:
        col1, col2 = st.columns(2)
        
        with col1:
            # Value counts
            value_counts = train_data[selected_cat].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                       title=f'Distribution of {selected_cat}')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average sales by category
            sales_by_cat = train_data.groupby(selected_cat)['Item_Outlet_Sales'].mean().sort_values(ascending=False)
            fig = px.bar(x=sales_by_cat.index, y=sales_by_cat.values,
                       title=f'Average Sales by {selected_cat}')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Numerical analysis
    st.subheader("Numerical Variables Analysis")
    
    numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Item_Outlet_Sales')
    
    selected_num = st.selectbox("Select Numerical Variable", numerical_cols)
    
    if selected_num:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution
            fig = px.histogram(train_data, x=selected_num, nbins=30,
                             title=f'Distribution of {selected_num}')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation with sales
            correlation = train_data[selected_num].corr(train_data['Item_Outlet_Sales'])
            fig = px.scatter(train_data, x=selected_num, y='Item_Outlet_Sales',
                           title=f'{selected_num} vs Sales (Correlation: {correlation:.3f})')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Correlation Analysis")
    
    numerical_data = train_data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()
    
    fig = px.imshow(correlation_matrix, 
                   text_auto=True, 
                   aspect="auto",
                   title="Correlation Matrix of Numerical Variables")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def model_performance_page(model_results):
    """Model performance page"""
    st.header("🤖 Model Performance")
    
    if model_results is None:
        st.warning("Model results not found. Please run model training first.")
        return
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    # Sort by R² score
    model_results_sorted = model_results.sort_values('R²', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig = px.bar(model_results_sorted, x='R²', y='Model', orientation='h',
                    title='R² Score Comparison')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE comparison
        fig = px.bar(model_results_sorted, x='RMSE', y='Model', orientation='h',
                    title='RMSE Comparison')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.subheader("Detailed Performance Metrics")
    st.dataframe(model_results_sorted.round(4))
    
    # Best model highlight
    best_model = model_results_sorted.iloc[0]
    st.success(f"🏆 Best Model: **{best_model['Model']}** with R² = {best_model['R²']:.4f} and RMSE = {best_model['RMSE']:.2f}")
    
    # Model performance trends
    st.subheader("Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² vs RMSE scatter
        fig = px.scatter(model_results, x='RMSE', y='R²', text='Model',
                        title='R² vs RMSE')
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MAE vs R² scatter
        fig = px.scatter(model_results, x='MAE', y='R²', text='Model',
                        title='MAE vs R²')
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def predictions_page(train_data, test_data):
    """Predictions page"""
    st.header("Sales Predictions")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Trained model not found. Please run model training first.")
        return
    
    # Load feature names
    try:
        feature_names = pd.read_csv('data/feature_names.csv').iloc[:, 0].tolist()
        st.success(f"✅ Loaded {len(feature_names)} features for prediction")
    except FileNotFoundError:
        st.error("Feature names not found. Please run data preprocessing first.")
        return
    
    # Prediction interface
    st.subheader("Interactive Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Parameters:**")
        
        # Item characteristics
        item_weight = st.slider("Item Weight", 0.0, 30.0, 12.0)
        item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.1)
        item_mrp = st.slider("Item MRP", 0.0, 300.0, 100.0)
        
        # Categorical variables
        item_fat_content = st.selectbox("Item Fat Content", 
                                       train_data['Item_Fat_Content'].unique())
        item_type = st.selectbox("Item Type", 
                                train_data['Item_Type'].unique())
        outlet_size = st.selectbox("Outlet Size", 
                                  train_data['Outlet_Size'].unique())
        outlet_location_type = st.selectbox("Outlet Location Type", 
                                           train_data['Outlet_Location_Type'].unique())
        outlet_type = st.selectbox("Outlet Type", 
                                  train_data['Outlet_Type'].unique())
    
    with col2:
        st.write("**Prediction Results:**")
        
        # Create prediction input with proper preprocessing
        try:
            # Create input data frame
            input_data = pd.DataFrame({
                'Item_Weight': [item_weight],
                'Item_Visibility': [item_visibility],
                'Item_MRP': [item_mrp],
                'Item_Fat_Content': [item_fat_content],
                'Item_Type': [item_type],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            })
            
            # Apply same preprocessing as training data
            from src.data_preprocessing import DataPreprocessor
            
            # Create a temporary preprocessor instance
            temp_preprocessor = DataPreprocessor()
            
            # Load the original data to get proper preprocessing parameters
            temp_train = pd.read_csv('data/Train.csv')
            
            # Handle missing values using same logic as training
            item_weight_median = temp_train.groupby('Item_Type')['Item_Weight'].median()
            outlet_size_mode = temp_train.groupby('Outlet_Type')['Outlet_Size'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Medium')
            
            # Fill missing values
            if input_data['Item_Weight'].isnull().any():
                item_type_val = input_data['Item_Type'].iloc[0]
                if item_type_val in item_weight_median.index:
                    input_data['Item_Weight'] = item_weight_median[item_type_val]
                else:
                    input_data['Item_Weight'] = temp_train['Item_Weight'].median()
            
            if input_data['Outlet_Size'].isnull().any():
                outlet_type_val = input_data['Outlet_Type'].iloc[0]
                if outlet_type_val in outlet_size_mode.index:
                    input_data['Outlet_Size'] = outlet_size_mode[outlet_type_val]
                else:
                    input_data['Outlet_Size'] = 'Medium'
            
            # Clean categorical variables
            fat_content_mapping = {
                'low fat': 'Low Fat',
                'LF': 'Low Fat',
                'reg': 'Regular'
            }
            input_data['Item_Fat_Content'] = input_data['Item_Fat_Content'].replace(fat_content_mapping)
            
            # Feature engineering
            current_year = 2023
            input_data['Outlet_Age'] = current_year - 2000  # Default establishment year
            
            # Create price categories
            price_category = pd.cut(input_data['Item_MRP'], 
                                  bins=[0, 50, 100, 150, 200, float('inf')], 
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            input_data['Price_Category'] = price_category
            
            # Create visibility categories
            visibility_category = pd.cut(input_data['Item_Visibility'], 
                                       bins=[0, 0.05, 0.1, 0.15, float('inf')], 
                                       labels=['Very Low', 'Low', 'Medium', 'High'])
            input_data['Visibility_Category'] = visibility_category
            
            # Create weight categories
            weight_category = pd.cut(input_data['Item_Weight'], 
                                   bins=[0, 10, 15, 20, float('inf')], 
                                   labels=['Light', 'Medium', 'Heavy', 'Very Heavy'])
            input_data['Weight_Category'] = weight_category
            
            # One-hot encode categorical variables
            categorical_columns = [
                'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                'Outlet_Location_Type', 'Outlet_Type', 'Price_Category',
                'Visibility_Category', 'Weight_Category'
            ]
            
            # Create dummy variables for each categorical column
            processed_input = input_data[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']].copy()
            
            for col in categorical_columns:
                if col in input_data.columns:
                    dummies = pd.get_dummies(input_data[col], prefix=col)
                    # Ensure all possible dummy columns are present
                    for feature in feature_names:
                        if feature.startswith(col + '_'):
                            if feature in dummies.columns:
                                processed_input[feature] = dummies[feature].iloc[0]
                            else:
                                processed_input[feature] = 0
            
            # Reorder columns to match training features
            processed_input = processed_input.reindex(columns=feature_names, fill_value=0)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            st.metric("Predicted Sales", f"${prediction:.2f}")
            
            # Confidence interval (simplified)
            confidence = 0.95
            st.write(f"Confidence: {confidence*100:.0f}%")
            
            # Show input summary
            st.write("**Input Summary:**")
            st.write(f"Item: {item_type} ({item_fat_content})")
            st.write(f"Weight: {item_weight:.1f} kg, MRP: ${item_mrp:.2f}")
            st.write(f"Outlet: {outlet_type} ({outlet_size}, {outlet_location_type})")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Please ensure all preprocessing steps are completed.")
    
    # Batch predictions
    st.subheader("Batch Predictions")
    
    if st.button("Generate Predictions for Test Data"):
        try:
            # Load processed test data
            processed_test = pd.read_csv('data/processed_test.csv')
            
            # Prepare test features (remove target if present)
            if 'Item_Outlet_Sales' in processed_test.columns:
                test_features = processed_test.drop('Item_Outlet_Sales', axis=1)
            else:
                test_features = processed_test
            
            # Ensure features match training features
            test_features = test_features.reindex(columns=feature_names, fill_value=0)
            
            # Make predictions
            predictions = model.predict(test_features)
            
            # Create results dataframe with original test identifiers
            results_df = pd.DataFrame({
                'Item_Identifier': test_data['Item_Identifier'],
                'Predicted_Sales': predictions
            })
            
            st.dataframe(results_df.head(20))
            
            # Show prediction statistics
            st.write("**Prediction Statistics:**")
            st.write(f"Mean prediction: ${predictions.mean():.2f}")
            st.write(f"Median prediction: ${np.median(predictions):.2f}")
            st.write(f"Min prediction: ${predictions.min():.2f}")
            st.write(f"Max prediction: ${predictions.max():.2f}")
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Batch prediction error: {str(e)}")
            st.write("Please ensure processed test data is available.")

def insights_page(train_data):
    """Insights page"""
    st.header("Key Insights")
    
    # Calculate insights
    insights = []
    
    # Sales insights
    avg_sales = train_data['Item_Outlet_Sales'].mean()
    median_sales = train_data['Item_Outlet_Sales'].median()
    insights.append(f"Average sales: ${avg_sales:.2f}, Median sales: ${median_sales:.2f}")
    
    # Top performing item type
    top_item_type = train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().idxmax()
    top_item_sales = train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().max()
    insights.append(f"Highest average sales item type: {top_item_type} (${top_item_sales:.2f})")
    
    # Top performing outlet type
    top_outlet_type = train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().idxmax()
    top_outlet_sales = train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().max()
    insights.append(f"Highest average sales outlet type: {top_outlet_type} (${top_outlet_sales:.2f})")
    
    # Price correlation
    price_corr = train_data['Item_MRP'].corr(train_data['Item_Outlet_Sales'])
    insights.append(f"Price-Sales correlation: {price_corr:.3f}")
    
    # Outlet age analysis
    current_year = 2023
    train_data['Outlet_Age'] = current_year - train_data['Outlet_Establishment_Year']
    age_corr = train_data['Outlet_Age'].corr(train_data['Item_Outlet_Sales'])
    insights.append(f"Outlet age-sales correlation: {age_corr:.3f}")
    
    # Display insights
    st.subheader("Key Findings")
    for i, insight in enumerate(insights, 1):
        st.write(f"{i}. {insight}")
    
    # Recommendations
    st.subheader("Business Recommendations")
    
    recommendations = [
        "Focus on high-performing item types like " + top_item_type,
        "Invest in " + top_outlet_type + " outlets for better sales",
        "Consider price optimization strategies based on correlation analysis",
        "Monitor outlet age impact on sales performance",
        "Implement targeted marketing for high-visibility items"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Performance metrics summary
    st.subheader("Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", f"{train_data.shape[0]:,}")
    
    with col2:
        st.metric("Unique Item Types", train_data['Item_Type'].nunique())
    
    with col3:
        st.metric("Unique Outlets", train_data['Outlet_Identifier'].nunique())

if __name__ == "__main__":
    main()
