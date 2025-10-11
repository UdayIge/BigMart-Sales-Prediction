"""
Test script to verify prediction functionality
Author: Data Analysis Project
"""

import pandas as pd
import pickle
import numpy as np

def test_prediction():
    """Test the prediction functionality"""
    print("🧪 Testing Prediction Functionality...")
    
    try:
        # Load the best model
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Load feature names
        feature_names = pd.read_csv('data/feature_names.csv').iloc[:, 0].tolist()
        print(f"✅ Loaded {len(feature_names)} feature names")
        
        # Load processed test data
        processed_test = pd.read_csv('data/processed_test.csv')
        print(f"✅ Loaded processed test data: {processed_test.shape}")
        
        # Prepare test features
        if 'Item_Outlet_Sales' in processed_test.columns:
            test_features = processed_test.drop('Item_Outlet_Sales', axis=1)
        else:
            test_features = processed_test
        
        # Ensure features match training features
        test_features = test_features.reindex(columns=feature_names, fill_value=0)
        print(f"✅ Prepared test features: {test_features.shape}")
        
        # Make predictions
        predictions = model.predict(test_features[:10])  # Test on first 10 samples
        print(f"✅ Generated predictions for 10 samples")
        
        # Display results
        print("\n📊 Sample Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: ${pred:.2f}")
        
        print(f"\n📈 Prediction Statistics:")
        print(f"Mean: ${predictions.mean():.2f}")
        print(f"Min: ${predictions.min():.2f}")
        print(f"Max: ${predictions.max():.2f}")
        
        print("\n🎉 Prediction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_prediction()
