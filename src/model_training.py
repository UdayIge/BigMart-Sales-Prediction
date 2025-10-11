"""
Model Training and Evaluation for BigMart Sales Prediction
Author: Data Analysis Project
Description: Comprehensive machine learning pipeline with multiple models and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import xgboost as xgb
from lightgbm import LGBMRegressor

class ModelTrainer:
    def __init__(self, train_path='data/processed_train.csv', test_path='data/processed_test.csv'):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load processed training and test data"""
        print("Loading processed data...")
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
        except FileNotFoundError:
            print("Processed data not found. Please run data preprocessing first.")
            return False
        return True
    
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
        """Initialize various machine learning models"""
        print("\n=== INITIALIZING MODELS ===")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': LGBMRegressor(random_state=42),
            'SVR': SVR(kernel='rbf'),
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
                if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                           'Elastic Net', 'SVR', 'KNN']:
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
        
        # Residual plot for best model
        plt.figure(figsize=(10, 6))
        residuals = self.y_test - y_pred_best
        plt.scatter(y_pred_best, residuals, alpha=0.5, color='orange')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot ({best_model_name})')
        plt.tight_layout()
        plt.savefig('results/residual_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Get the best model name
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'R²': [self.results[name]['r2'] for name in self.results.keys()]
        })
        best_model_name = results_df.sort_values('R²', ascending=False).iloc[0]['Model']
        
        print(f"Performing hyperparameter tuning for {best_model_name}")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if best_model_name in param_grids:
            # Perform grid search
            grid_search = GridSearchCV(
                self.results[best_model_name]['model'],
                param_grids[best_model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Evaluate tuned model
            y_pred_tuned = self.best_model.predict(self.X_test)
            r2_tuned = r2_score(self.y_test, y_pred_tuned)
            rmse_tuned = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
            
            print(f"Tuned model R²: {r2_tuned:.4f}")
            print(f"Tuned model RMSE: {rmse_tuned:.2f}")
            
            # Save tuned model
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            print("Tuned model saved to 'models/best_model.pkl'")
            
        else:
            print(f"Hyperparameter tuning not implemented for {best_model_name}")
            
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature names
        feature_names = self.X_train.columns.tolist()
        
        # Analyze feature importance for tree-based models
        tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        
        for model_name in tree_models:
            if model_name in self.results:
                try:
                    model = self.results[model_name]['model']
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        
                        # Create feature importance DataFrame
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot top 20 features
                        plt.figure(figsize=(12, 8))
                        top_features = importance_df.head(20)
                        plt.barh(range(len(top_features)), top_features['Importance'])
                        plt.yticks(range(len(top_features)), top_features['Feature'])
                        plt.xlabel('Feature Importance')
                        plt.title(f'Top 20 Feature Importance - {model_name}')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        plt.savefig(f'results/feature_importance_{model_name.replace(" ", "_")}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.show()
                        
                        # Save feature importance
                        importance_df.to_csv(f'results/feature_importance_{model_name.replace(" ", "_")}.csv', 
                                           index=False)
                        
                        print(f"\nTop 10 features for {model_name}:")
                        print(importance_df.head(10))
                        
                except Exception as e:
                    print(f"Error analyzing feature importance for {model_name}: {str(e)}")
                    
    def generate_predictions(self):
        """Generate predictions on test data"""
        print("\n=== GENERATING PREDICTIONS ===")
        
        if self.best_model is None:
            print("No best model found. Please train models first.")
            return
        
        # Generate predictions on test data
        test_predictions = self.best_model.predict(self.test_data)
        
        # Create submission file
        submission = pd.DataFrame({
            'Item_Identifier': self.test_data.index,
            'Item_Outlet_Sales': test_predictions
        })
        
        submission.to_csv('results/submission.csv', index=False)
        print("Predictions saved to 'results/submission.csv'")
        
        # Display prediction statistics
        print(f"\nPrediction Statistics:")
        print(f"Mean prediction: ${test_predictions.mean():.2f}")
        print(f"Median prediction: ${np.median(test_predictions):.2f}")
        print(f"Min prediction: ${test_predictions.min():.2f}")
        print(f"Max prediction: ${test_predictions.max():.2f}")
        print(f"Standard deviation: ${test_predictions.std():.2f}")
        
        return test_predictions
        
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
        
    def run_full_training(self):
        """Run the complete model training pipeline"""
        print("=== STARTING MODEL TRAINING PIPELINE ===")
        
        # Load data
        if not self.load_data():
            return
        
        # Prepare data
        self.prepare_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Visualize results
        self.visualize_results()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Feature importance analysis
        self.feature_importance_analysis()
        
        # Generate predictions
        self.generate_predictions()
        
        # Save models
        self.save_models()
        
        print("\n=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
        
        # Final summary
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'R²': [self.results[name]['r2'] for name in self.results.keys()],
            'RMSE': [self.results[name]['rmse'] for name in self.results.keys()]
        }).sort_values('R²', ascending=False)
        
        print("\nFinal Model Rankings:")
        print(results_df.head())
        
        return results_df

def main():
    """Main function to run model training"""
    trainer = ModelTrainer()
    results = trainer.run_full_training()
    return results

if __name__ == "__main__":
    main()
