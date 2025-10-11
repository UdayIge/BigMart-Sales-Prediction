"""
Simplified Model Training for BigMart Sales Prediction
Author: Data Analysis Project
Description: Basic model training without advanced ML libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimpleModelTrainer:
    def __init__(self, train_path='data/processed_train.csv'):
        self.train_path = train_path
        self.train_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load processed training data"""
        print("Loading processed data...")
        try:
            self.train_data = pd.read_csv(self.train_path)
            print(f"Training data shape: {self.train_data.shape}")
            return True
        except FileNotFoundError:
            print("Processed data not found. Please run data preprocessing first.")
            return False
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n=== PREPARING DATA FOR TRAINING ===")
        
        # Separate features and target
        X = self.train_data.drop('Item_Outlet_Sales', axis=1)
        y = self.train_data['Item_Outlet_Sales']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preparation completed!")
        
    def initialize_models(self):
        """Initialize basic machine learning models"""
        print("\n=== INITIALIZING MODELS ===")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} models")
        
    def train_models(self):
        """Train all models and evaluate performance"""
        print("\n=== TRAINING MODELS ===")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for models that benefit from it
                if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
                
                self.results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        print(f"\nSuccessfully trained {len(self.results)} models")
        
    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\n=== MODEL EVALUATION ===")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'RMSE': [self.results[name]['rmse'] for name in self.results.keys()],
            'MAE': [self.results[name]['mae'] for name in self.results.keys()],
            'R²': [self.results[name]['r2'] for name in self.results.keys()],
            'CV Mean': [self.results[name]['cv_mean'] for name in self.results.keys()],
            'CV Std': [self.results[name]['cv_std'] for name in self.results.keys()]
        })
        
        # Sort by R² score
        results_df = results_df.sort_values('R²', ascending=False)
        
        print("Model Performance Comparison:")
        print(results_df.round(4))
        
        # Save results
        results_df.to_csv('results/model_performance.csv', index=False)
        print("\nResults saved to 'results/model_performance.csv'")
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best R² Score: {results_df.iloc[0]['R²']:.4f}")
        print(f"Best RMSE: {results_df.iloc[0]['RMSE']:.2f}")
        
        return results_df
        
    def visualize_results(self):
        """Create visualizations for model comparison"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Model performance comparison
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'RMSE': [self.results[name]['rmse'] for name in self.results.keys()],
            'MAE': [self.results[name]['mae'] for name in self.results.keys()],
            'R²': [self.results[name]['r2'] for name in self.results.keys()]
        })
        
        # Sort by R² score
        results_df = results_df.sort_values('R²', ascending=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        axes[0, 0].barh(results_df['Model'], results_df['R²'], color='skyblue')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xlabel('R² Score')
        
        # RMSE comparison
        axes[0, 1].barh(results_df['Model'], results_df['RMSE'], color='lightcoral')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_xlabel('RMSE')
        
        # MAE comparison
        axes[1, 0].barh(results_df['Model'], results_df['MAE'], color='lightgreen')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_xlabel('MAE')
        
        # Actual vs Predicted for best model
        best_model_name = results_df.iloc[-1]['Model']
        y_pred_best = self.results[best_model_name]['predictions']
        
        axes[1, 1].scatter(self.y_test, y_pred_best, alpha=0.5, color='purple')
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Sales')
        axes[1, 1].set_ylabel('Predicted Sales')
        axes[1, 1].set_title(f'Actual vs Predicted ({best_model_name})')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_models(self):
        """Save all trained models"""
        print("\n=== SAVING MODELS ===")
        
        for name, result in self.results.items():
            try:
                model_filename = f"models/{name.replace(' ', '_').lower()}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(result['model'], f)
                print(f"Saved {name} to {model_filename}")
            except Exception as e:
                print(f"Error saving {name}: {str(e)}")
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Saved scaler to models/scaler.pkl")
        
        # Save best model
        if self.best_model:
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            print("Saved best model to models/best_model.pkl")
        
    def run_training(self):
        """Run the complete model training pipeline"""
        print("=== STARTING SIMPLIFIED MODEL TRAINING ===")
        
        # Load data
        if not self.load_data():
            return None
        
        # Prepare data
        self.prepare_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        results_df = self.evaluate_models()
        
        # Visualize results
        self.visualize_results()
        
        # Save models
        self.save_models()
        
        print("\n=== MODEL TRAINING COMPLETED ===")
        
        # Final summary
        print("\nFinal Model Rankings:")
        print(results_df.head())
        
        return results_df

def main():
    """Main function to run simplified model training"""
    trainer = SimpleModelTrainer()
    results = trainer.run_training()
    return results

if __name__ == "__main__":
    main()
