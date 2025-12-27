import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os

class FuelPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_columns = []
        
    def load_and_train(self, dataset_path='fuel_predictions_india.xls'):
        """Load dataset and train the model"""
        try:
            # Load the dataset
            df = pd.read_excel(dataset_path)
            
            # Select reasonable numeric features for fuel prediction
            # Assuming common features that might affect fuel consumption
            possible_features = [
                'distance', 'speed', 'temperature', 'weight', 
                'time', 'elevation_change', 'traffic_density'
            ]
            
            # Select only the columns that exist in the dataset
            self.feature_columns = [col for col in possible_features if col in df.columns]
            
            # If specific features aren't found, use all numeric columns
            if not self.feature_columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove the target variable if it exists in numeric columns
                if 'fuel_consumed' in numeric_cols:
                    numeric_cols.remove('fuel_consumed')
                self.feature_columns = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            
            if not self.feature_columns:
                raise ValueError("No suitable features found in the dataset")
                
            X = df[self.feature_columns].fillna(0)  # Fill any missing values
            y = df['fuel_consumed'] if 'fuel_consumed' in df.columns else df.iloc[:, 0]  # Assume first column is target if fuel_consumed not found
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            
        except Exception as e:
            print(f"Error loading or training model: {e}")
            self.is_trained = False
    
    def predict_fuel(self, features_dict):
        """Predict fuel consumption based on input features"""
        try:
            if not self.is_trained:
                # Fallback: return distance / average mileage (5 km/liter as default)
                distance = features_dict.get('distance', 10)
                return distance / 5  # Default assumption: 5 km per liter
            
            # Prepare input for prediction
            input_features = []
            for col in self.feature_columns:
                input_features.append(features_dict.get(col, 0))
            
            # Reshape for sklearn
            input_array = np.array(input_features).reshape(1, -1)
            
            # Make prediction
            predicted_fuel = self.model.predict(input_array)[0]
            
            # Ensure prediction is non-negative
            return max(0, predicted_fuel)
            
        except Exception as e:
            # Fallback: calculate based on distance and average mileage
            distance = features_dict.get('distance', 10)
            # Assuming average mileage of 5 km/liter as default
            return distance / 5

# Global instance
fuel_predictor = FuelPredictor()

def predict_fuel(features_dict):
    """
    Predict fuel consumption in liters based on input features.
    
    Args:
        features_dict (dict): Dictionary containing features for prediction
        
    Returns:
        float: Predicted fuel consumption in liters
    """
    return fuel_predictor.predict_fuel(features_dict)

# Initialize the model when module is loaded
if os.path.exists('fuel_predictions_india.xls'):
    fuel_predictor.load_and_train('fuel_predictions_india.xls')