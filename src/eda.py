"""
Exploratory Data Analysis (EDA) for BigMart Sales Prediction
Author: Data Analysis Project
Description: Comprehensive exploratory data analysis with visualizations and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EDAAnalyzer:
    def __init__(self, train_path='data/Train.csv', processed_train_path='data/processed_train.csv'):
        self.train_data = None
        self.processed_train = None
        self.train_path = train_path
        self.processed_train_path = processed_train_path
        
    def load_data(self):
        """Load original and processed training data"""
        print("Loading data for EDA...")
        self.train_data = pd.read_csv(self.train_path)
        
        try:
            self.processed_train = pd.read_csv(self.processed_train_path)
            print("Both original and processed data loaded successfully!")
        except FileNotFoundError:
            print("Processed data not found. Running with original data only.")
            self.processed_train = None
            
        print(f"Original data shape: {self.train_data.shape}")
        if self.processed_train is not None:
            print(f"Processed data shape: {self.processed_train.shape}")
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        
        print(f"Dataset Shape: {self.train_data.shape}")
        print(f"Number of Rows: {self.train_data.shape[0]:,}")
        print(f"Number of Columns: {self.train_data.shape[1]}")
        
        print("\nColumn Names and Data Types:")
        print(self.train_data.dtypes)
        
        print("\nFirst 5 rows:")
        print(self.train_data.head())
        
        print("\nLast 5 rows:")
        print(self.train_data.tail())
        
        print("\nDataset Info:")
        self.train_data.info()
        
    def missing_values_analysis(self):
        """Analyze missing values in the dataset"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing_data = self.train_data.isnull().sum()
        missing_percentage = (missing_data / len(self.train_data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percentage
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print("Missing Values Summary:")
            print(missing_df)
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            missing_df['Missing Percentage'].plot(kind='bar')
            plt.title('Missing Values Percentage by Column')
            plt.xlabel('Columns')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No missing values found in the dataset!")
    
    def target_variable_analysis(self):
        """Analyze the target variable (Item_Outlet_Sales)"""
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        target = self.train_data['Item_Outlet_Sales']
        
        print("Target Variable Statistics:")
        print(f"Mean: ${target.mean():.2f}")
        print(f"Median: ${target.median():.2f}")
        print(f"Standard Deviation: ${target.std():.2f}")
        print(f"Minimum: ${target.min():.2f}")
        print(f"Maximum: ${target.max():.2f}")
        print(f"Skewness: {target.skew():.2f}")
        print(f"Kurtosis: {target.kurtosis():.2f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        axes[0, 0].hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Item Outlet Sales')
        axes[0, 0].set_xlabel('Sales ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(target)
        axes[0, 1].set_title('Box Plot of Item Outlet Sales')
        axes[0, 1].set_ylabel('Sales ($)')
        
        # Log transformation
        log_target = np.log1p(target)
        axes[1, 0].hist(log_target, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Distribution of Log-Transformed Sales')
        axes[1, 0].set_xlabel('Log(Sales)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Sales')
        
        plt.tight_layout()
        plt.savefig('results/target_variable_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        print("\n" + "="*50)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("="*50)
        
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('Item_Identifier')  # Remove identifier
        
        print(f"Categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            print(f"\n--- {col} ---")
            value_counts = self.train_data[col].value_counts()
            print(f"Unique values: {self.train_data[col].nunique()}")
            print("Value counts:")
            print(value_counts.head(10))
            
            # Create visualizations for each categorical variable
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Count plot
            value_counts.plot(kind='bar', ax=axes[0])
            axes[0].set_title(f'Distribution of {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Sales by category
            sales_by_category = self.train_data.groupby(col)['Item_Outlet_Sales'].mean().sort_values(ascending=False)
            sales_by_category.plot(kind='bar', ax=axes[1], color='orange')
            axes[1].set_title(f'Average Sales by {col}')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Average Sales ($)')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'results/categorical_analysis_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def numerical_analysis(self):
        """Analyze numerical variables"""
        print("\n" + "="*50)
        print("NUMERICAL VARIABLES ANALYSIS")
        print("="*50)
        
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove('Item_Outlet_Sales')  # Remove target variable
        
        print(f"Numerical columns: {numerical_cols}")
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.train_data[numerical_cols].describe())
        
        # Create visualizations
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numerical_cols):
            # Histogram
            axes[i, 0].hist(self.train_data[col], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[i, 0].set_title(f'Distribution of {col}')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[i, 1].boxplot(self.train_data[col])
            axes[i, 1].set_title(f'Box Plot of {col}')
            axes[i, 1].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('results/numerical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Select numerical columns for correlation
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = self.train_data[numerical_cols].corr()
        
        print("Correlation Matrix:")
        print(correlation_matrix)
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation with target variable
        target_corr = correlation_matrix['Item_Outlet_Sales'].drop('Item_Outlet_Sales').sort_values(key=abs, ascending=False)
        print("\nCorrelation with Target Variable (Item_Outlet_Sales):")
        print(target_corr)
        
        # Visualize correlation with target
        plt.figure(figsize=(10, 6))
        target_corr.plot(kind='bar', color='purple')
        plt.title('Correlation with Item Outlet Sales')
        plt.xlabel('Variables')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/target_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def sales_pattern_analysis(self):
        """Analyze sales patterns across different dimensions"""
        print("\n" + "="*50)
        print("SALES PATTERN ANALYSIS")
        print("="*50)
        
        # Sales by Item Type
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Sales by Item Type
        plt.subplot(2, 2, 1)
        sales_by_item_type = self.train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
        sales_by_item_type.plot(kind='bar', color='skyblue')
        plt.title('Average Sales by Item Type')
        plt.xlabel('Item Type')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        # Subplot 2: Sales by Outlet Type
        plt.subplot(2, 2, 2)
        sales_by_outlet_type = self.train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
        sales_by_outlet_type.plot(kind='bar', color='lightgreen')
        plt.title('Average Sales by Outlet Type')
        plt.xlabel('Outlet Type')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        # Subplot 3: Sales by Outlet Size
        plt.subplot(2, 2, 3)
        sales_by_outlet_size = self.train_data.groupby('Outlet_Size')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
        sales_by_outlet_size.plot(kind='bar', color='orange')
        plt.title('Average Sales by Outlet Size')
        plt.xlabel('Outlet Size')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        # Subplot 4: Sales by Outlet Location Type
        plt.subplot(2, 2, 4)
        sales_by_location = self.train_data.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
        sales_by_location.plot(kind='bar', color='pink')
        plt.title('Average Sales by Outlet Location Type')
        plt.xlabel('Outlet Location Type')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/sales_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed analysis
        print("\nTop 5 Item Types by Average Sales:")
        print(sales_by_item_type.head())
        
        print("\nTop 5 Outlet Types by Average Sales:")
        print(sales_by_outlet_type.head())
    
    def price_analysis(self):
        """Analyze price patterns and their relationship with sales"""
        print("\n" + "="*50)
        print("PRICE ANALYSIS")
        print("="*50)
        
        # Price vs Sales scatter plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(self.train_data['Item_MRP'], self.train_data['Item_Outlet_Sales'], alpha=0.5, color='blue')
        plt.xlabel('Item MRP ($)')
        plt.ylabel('Item Outlet Sales ($)')
        plt.title('Price vs Sales Scatter Plot')
        
        # Price distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.train_data['Item_MRP'], bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Item MRP ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Item MRP')
        
        # Sales by price range
        plt.subplot(2, 2, 3)
        price_bins = pd.cut(self.train_data['Item_MRP'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        sales_by_price = self.train_data.groupby(price_bins)['Item_Outlet_Sales'].mean()
        sales_by_price.plot(kind='bar', color='red')
        plt.title('Average Sales by Price Range')
        plt.xlabel('Price Range')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        # Correlation analysis
        plt.subplot(2, 2, 4)
        correlation = self.train_data['Item_MRP'].corr(self.train_data['Item_Outlet_Sales'])
        plt.text(0.5, 0.5, f'Correlation: {correlation:.3f}', ha='center', va='center', fontsize=16)
        plt.title('Price-Sales Correlation')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Price-Sales Correlation: {correlation:.3f}")
    
    def outlet_analysis(self):
        """Analyze outlet characteristics and performance"""
        print("\n" + "="*50)
        print("OUTLET ANALYSIS")
        print("="*50)
        
        # Outlet performance analysis
        outlet_stats = self.train_data.groupby('Outlet_Identifier').agg({
            'Item_Outlet_Sales': ['count', 'mean', 'sum', 'std'],
            'Item_MRP': 'mean',
            'Outlet_Establishment_Year': 'first',
            'Outlet_Size': 'first',
            'Outlet_Location_Type': 'first',
            'Outlet_Type': 'first'
        }).round(2)
        
        outlet_stats.columns = ['Item_Count', 'Avg_Sales', 'Total_Sales', 'Sales_Std', 'Avg_MRP', 'Est_Year', 'Size', 'Location', 'Type']
        outlet_stats = outlet_stats.sort_values('Total_Sales', ascending=False)
        
        print("Outlet Performance Summary:")
        print(outlet_stats.head(10))
        
        # Visualize outlet performance
        plt.figure(figsize=(15, 10))
        
        # Total sales by outlet
        plt.subplot(2, 2, 1)
        outlet_stats['Total_Sales'].head(10).plot(kind='bar', color='purple')
        plt.title('Top 10 Outlets by Total Sales')
        plt.xlabel('Outlet Identifier')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        
        # Average sales by outlet
        plt.subplot(2, 2, 2)
        outlet_stats['Avg_Sales'].head(10).plot(kind='bar', color='orange')
        plt.title('Top 10 Outlets by Average Sales')
        plt.xlabel('Outlet Identifier')
        plt.ylabel('Average Sales ($)')
        plt.xticks(rotation=45)
        
        # Sales vs establishment year
        plt.subplot(2, 2, 3)
        plt.scatter(outlet_stats['Est_Year'], outlet_stats['Avg_Sales'], alpha=0.7, color='green')
        plt.xlabel('Establishment Year')
        plt.ylabel('Average Sales ($)')
        plt.title('Sales vs Outlet Age')
        
        # Item count vs sales
        plt.subplot(2, 2, 4)
        plt.scatter(outlet_stats['Item_Count'], outlet_stats['Total_Sales'], alpha=0.7, color='red')
        plt.xlabel('Number of Items')
        plt.ylabel('Total Sales ($)')
        plt.title('Item Count vs Total Sales')
        
        plt.tight_layout()
        plt.savefig('results/outlet_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outlet_stats
    
    def generate_insights(self):
        """Generate key insights from the EDA"""
        print("\n" + "="*50)
        print("KEY INSIGHTS FROM EDA")
        print("="*50)
        
        insights = []
        
        # Target variable insights
        target_stats = self.train_data['Item_Outlet_Sales'].describe()
        insights.append(f"Sales range from ${target_stats['min']:.2f} to ${target_stats['max']:.2f} with mean ${target_stats['mean']:.2f}")
        
        # Top performing item type
        top_item_type = self.train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().idxmax()
        top_item_sales = self.train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().max()
        insights.append(f"Highest average sales item type: {top_item_type} (${top_item_sales:.2f})")
        
        # Top performing outlet type
        top_outlet_type = self.train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().idxmax()
        top_outlet_sales = self.train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().max()
        insights.append(f"Highest average sales outlet type: {top_outlet_type} (${top_outlet_sales:.2f})")
        
        # Price correlation
        price_corr = self.train_data['Item_MRP'].corr(self.train_data['Item_Outlet_Sales'])
        insights.append(f"Price-Sales correlation: {price_corr:.3f} ({'strong' if abs(price_corr) > 0.7 else 'moderate' if abs(price_corr) > 0.3 else 'weak'} relationship)")
        
        # Missing values
        missing_items = self.train_data['Item_Weight'].isnull().sum()
        missing_outlets = self.train_data['Outlet_Size'].isnull().sum()
        insights.append(f"Missing values: {missing_items} item weights, {missing_outlets} outlet sizes")
        
        # Outlet age analysis
        current_year = 2023
        self.train_data['Outlet_Age'] = current_year - self.train_data['Outlet_Establishment_Year']
        age_corr = self.train_data['Outlet_Age'].corr(self.train_data['Item_Outlet_Sales'])
        insights.append(f"Outlet age-sales correlation: {age_corr:.3f}")
        
        print("Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
    
    def create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        print("\n" + "="*50)
        print("CREATING INTERACTIVE PLOTS")
        print("="*50)
        
        # Interactive scatter plot: Price vs Sales (handle NaN values)
        # Create a copy of data and fill NaN values for plotting
        plot_data = self.train_data.copy()
        plot_data['Item_Weight'] = plot_data['Item_Weight'].fillna(plot_data['Item_Weight'].median())
        
        fig = px.scatter(plot_data, x='Item_MRP', y='Item_Outlet_Sales', 
                        color='Item_Type', size='Item_Weight',
                        title='Interactive: Price vs Sales by Item Type',
                        labels={'Item_MRP': 'Item MRP ($)', 'Item_Outlet_Sales': 'Item Outlet Sales ($)'})
        fig.write_html('results/interactive_price_sales.html')
        
        # Interactive box plot: Sales by Outlet Type
        fig = px.box(self.train_data, x='Outlet_Type', y='Item_Outlet_Sales',
                    title='Interactive: Sales Distribution by Outlet Type',
                    labels={'Outlet_Type': 'Outlet Type', 'Item_Outlet_Sales': 'Item Outlet Sales ($)'})
        fig.write_html('results/interactive_outlet_sales.html')
        
        # Interactive bar chart: Average Sales by Item Type
        sales_by_item = self.train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().reset_index()
        fig = px.bar(sales_by_item, x='Item_Type', y='Item_Outlet_Sales',
                    title='Interactive: Average Sales by Item Type',
                    labels={'Item_Type': 'Item Type', 'Item_Outlet_Sales': 'Average Sales ($)'})
        fig.update_xaxes(tickangle=45)
        fig.write_html('results/interactive_item_sales.html')
        
        print("Interactive plots saved to results/ directory")
    
    def run_full_eda(self):
        """Run the complete EDA pipeline"""
        print("=== STARTING EXPLORATORY DATA ANALYSIS ===")
        
        # Load data
        self.load_data()
        
        # Basic information
        self.basic_info()
        
        # Missing values analysis
        self.missing_values_analysis()
        
        # Target variable analysis
        self.target_variable_analysis()
        
        # Categorical analysis
        self.categorical_analysis()
        
        # Numerical analysis
        self.numerical_analysis()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Sales pattern analysis
        self.sales_pattern_analysis()
        
        # Price analysis
        self.price_analysis()
        
        # Outlet analysis
        self.outlet_analysis()
        
        # Generate insights
        self.generate_insights()
        
        # Create interactive plots
        self.create_interactive_plots()
        
        print("\n=== EDA COMPLETED SUCCESSFULLY ===")
        print("All visualizations and analysis saved to 'results/' directory")

def main():
    """Main function to run EDA"""
    eda = EDAAnalyzer()
    eda.run_full_eda()

if __name__ == "__main__":
    main()
