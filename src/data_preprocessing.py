"""
Data Preprocessing Pipeline for BigMart Sales Prediction
Author: Data Analysis Project
Description: Comprehensive data cleaning and feature engineering for sales prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.processed_train = None
        self.processed_test = None
        
    def load_data(self, train_path='data/Train.csv', test_path='data/Test.csv'):
        """Load training and test datasets"""
        print("Loading datasets...")
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print("\nTraining Data Info:")
        print(self.train_data.info())
        
        print("\nTest Data Info:")
        print(self.test_data.info())
        
        print("\nMissing Values in Training Data:")
        print(self.train_data.isnull().sum())
        
        print("\nMissing Values in Test Data:")
        print(self.test_data.isnull().sum())
        
        print("\nTraining Data Description:")
        print(self.train_data.describe())
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # For Item_Weight: Fill with median weight of the same item type
        print("Filling missing Item_Weight values...")
        
        # Training data
        item_weight_median = self.train_data.groupby('Item_Type')['Item_Weight'].median()
        for item_type in self.train_data['Item_Type'].unique():
            mask = (self.train_data['Item_Type'] == item_type) & (self.train_data['Item_Weight'].isnull())
            self.train_data.loc[mask, 'Item_Weight'] = item_weight_median[item_type]
        
        # Test data
        for item_type in self.test_data['Item_Type'].unique():
            mask = (self.test_data['Item_Type'] == item_type) & (self.test_data['Item_Weight'].isnull())
            if item_type in item_weight_median.index:
                self.test_data.loc[mask, 'Item_Weight'] = item_weight_median[item_type]
            else:
                # If item type not in training data, use overall median
                self.test_data.loc[mask, 'Item_Weight'] = self.train_data['Item_Weight'].median()
        
        # For Outlet_Size: Fill with mode (most frequent) size for the same outlet type
        print("Filling missing Outlet_Size values...")
        
        # Training data
        outlet_size_mode = self.train_data.groupby('Outlet_Type')['Outlet_Size'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Medium')
        for outlet_type in self.train_data['Outlet_Type'].unique():
            mask = (self.train_data['Outlet_Type'] == outlet_type) & (self.train_data['Outlet_Size'].isnull())
            self.train_data.loc[mask, 'Outlet_Size'] = outlet_size_mode[outlet_type]
        
        # Test data
        for outlet_type in self.test_data['Outlet_Type'].unique():
            mask = (self.test_data['Outlet_Type'] == outlet_type) & (self.test_data['Outlet_Size'].isnull())
            if outlet_type in outlet_size_mode.index:
                self.test_data.loc[mask, 'Outlet_Size'] = outlet_size_mode[outlet_type]
            else:
                self.test_data.loc[mask, 'Outlet_Size'] = 'Medium'  # Default
        
        print("Missing values handled successfully!")
        
    def clean_categorical_variables(self):
        """Clean and standardize categorical variables"""
        print("\n=== CLEANING CATEGORICAL VARIABLES ===")
        
        # Fix Item_Fat_Content inconsistencies
        fat_content_mapping = {
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        }
        
        self.train_data['Item_Fat_Content'] = self.train_data['Item_Fat_Content'].replace(fat_content_mapping)
        self.test_data['Item_Fat_Content'] = self.test_data['Item_Fat_Content'].replace(fat_content_mapping)
        
        print("Categorical variables cleaned!")
        
    def feature_engineering(self):
        """Create new features for better model performance"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Calculate outlet age
        current_year = 2023
        self.train_data['Outlet_Age'] = current_year - self.train_data['Outlet_Establishment_Year']
        self.test_data['Outlet_Age'] = current_year - self.test_data['Outlet_Establishment_Year']
        
        # Create price categories based on MRP
        self.train_data['Price_Category'] = pd.cut(
            self.train_data['Item_MRP'], 
            bins=[0, 50, 100, 150, 200, float('inf')], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        self.test_data['Price_Category'] = pd.cut(
            self.test_data['Item_MRP'], 
            bins=[0, 50, 100, 150, 200, float('inf')], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Create visibility categories
        self.train_data['Visibility_Category'] = pd.cut(
            self.train_data['Item_Visibility'], 
            bins=[0, 0.05, 0.1, 0.15, float('inf')], 
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
        
        self.test_data['Visibility_Category'] = pd.cut(
            self.test_data['Item_Visibility'], 
            bins=[0, 0.05, 0.1, 0.15, float('inf')], 
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
        
        # Create weight categories
        self.train_data['Weight_Category'] = pd.cut(
            self.train_data['Item_Weight'], 
            bins=[0, 10, 15, 20, float('inf')], 
            labels=['Light', 'Medium', 'Heavy', 'Very Heavy']
        )
        
        self.test_data['Weight_Category'] = pd.cut(
            self.test_data['Item_Weight'], 
            bins=[0, 10, 15, 20, float('inf')], 
            labels=['Light', 'Medium', 'Heavy', 'Very Heavy']
        )
        
        print("New features created successfully!")
        
    def encode_categorical_variables(self):
        """Encode categorical variables for machine learning"""
        print("\n=== ENCODING CATEGORICAL VARIABLES ===")
        
        # List of categorical columns to encode
        categorical_columns = [
            'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
            'Outlet_Location_Type', 'Outlet_Type', 'Price_Category',
            'Visibility_Category', 'Weight_Category'
        ]
        
        # Create processed datasets
        self.processed_train = self.train_data.copy()
        self.processed_test = self.test_data.copy()
        
        # One-hot encode categorical variables
        for col in categorical_columns:
            if col in self.processed_train.columns:
                # Get unique values from both train and test
                unique_values = set(self.processed_train[col].unique()) | set(self.processed_test[col].unique())
                
                # Create dummy variables
                train_dummies = pd.get_dummies(self.processed_train[col], prefix=col)
                test_dummies = pd.get_dummies(self.processed_test[col], prefix=col)
                
                # Ensure both have the same columns
                for val in unique_values:
                    col_name = f"{col}_{val}"
                    if col_name not in train_dummies.columns:
                        train_dummies[col_name] = 0
                    if col_name not in test_dummies.columns:
                        test_dummies[col_name] = 0
                
                # Drop original column and add dummy columns
                self.processed_train = self.processed_train.drop(col, axis=1)
                self.processed_test = self.processed_test.drop(col, axis=1)
                
                self.processed_train = pd.concat([self.processed_train, train_dummies], axis=1)
                self.processed_test = pd.concat([self.processed_test, test_dummies], axis=1)
        
        # Drop identifier columns (not needed for modeling)
        columns_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year']
        
        for col in columns_to_drop:
            if col in self.processed_train.columns:
                self.processed_train = self.processed_train.drop(col, axis=1)
            if col in self.processed_test.columns:
                self.processed_test = self.processed_test.drop(col, axis=1)
        
        print("Categorical variables encoded successfully!")
        
    def save_processed_data(self):
        """Save processed datasets"""
        print("\n=== SAVING PROCESSED DATA ===")
        
        # Save processed training data
        self.processed_train.to_csv('data/processed_train.csv', index=False)
        print("Processed training data saved to 'data/processed_train.csv'")
        
        # Save processed test data
        self.processed_test.to_csv('data/processed_test.csv', index=False)
        print("Processed test data saved to 'data/processed_test.csv'")
        
        # Save feature names for later use
        feature_names = [col for col in self.processed_train.columns if col != 'Item_Outlet_Sales']
        pd.Series(feature_names).to_csv('data/feature_names.csv', index=False)
        print("Feature names saved to 'data/feature_names.csv'")
        
    def get_processed_data(self):
        """Return processed datasets"""
        return self.processed_train, self.processed_test
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("=== STARTING DATA PREPROCESSING PIPELINE ===")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Clean categorical variables
        self.clean_categorical_variables()
        
        # Feature engineering
        self.feature_engineering()
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Save processed data
        self.save_processed_data()
        
        print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        print(f"Final training data shape: {self.processed_train.shape}")
        print(f"Final test data shape: {self.processed_test.shape}")
        
        return self.processed_train, self.processed_test

def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.run_full_preprocessing()
    
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Training data: {train_processed.shape[0]} rows, {train_processed.shape[1]} columns")
    print(f"Test data: {test_processed.shape[0]} rows, {test_processed.shape[1]} columns")
    print(f"Target variable: Item_Outlet_Sales")
    print(f"Features: {train_processed.shape[1] - 1}")

if __name__ == "__main__":
    main()
