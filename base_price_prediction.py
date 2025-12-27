import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# =========================
# 1. Create Synthetic Dataset
# =========================
import numpy as np

# Generate synthetic data that matches the expected format
np.random.seed(42)
num_samples = 1000

df = pd.DataFrame({
    "pickup_region": np.random.choice(["Maharashtra", "Tamil Nadu", "Karnataka", "Gujarat", "Delhi", "Uttar Pradesh", "Kerala", "Punjab"], num_samples),
    "drop_region": np.random.choice(["Maharashtra", "Tamil Nadu", "Karnataka", "Gujarat", "Delhi", "Uttar Pradesh", "Kerala", "Punjab"], num_samples),
    "cargo_type": np.random.choice(["Electronics", "Textiles", "Food Items", "Machinery", "Chemicals", "Automotive", "Pharmaceuticals", "Furniture"], num_samples),
    "distance_km": np.random.uniform(50, 2000, num_samples),  # Distance between 50 and 2000 km
    "cargo_weight_kg": np.random.uniform(100, 5000, num_samples),  # Weight between 100 and 5000 kg
    "cargo_volume_m3": np.random.uniform(0.5, 20, num_samples),  # Volume between 0.5 and 20 cubic meters
    "cargo_value_inr": np.random.uniform(10000, 1000000, num_samples),  # Value between 10,000 and 1,000,000 INR
})


# Calculate base_price_inr based on realistic factors
# Price increases with distance, weight, value, and certain cargo types
base_prices = []
for i in range(num_samples):
    base_price = (
        500 +  # Base cost
        df.iloc[i]["distance_km"] * 2 +  # Cost per km
        df.iloc[i]["cargo_weight_kg"] * 0.5 +  # Cost per kg
        df.iloc[i]["cargo_volume_m3"] * 100 +  # Cost per cubic meter
        df.iloc[i]["cargo_value_inr"] * 0.001  # Cost based on value
    )
    
    # Adjust for cargo type risk
    if df.iloc[i]["cargo_type"] in ["Electronics", "Pharmaceuticals"]:
        base_price *= 1.3  # Higher cost for sensitive cargo
    elif df.iloc[i]["cargo_type"] in ["Food Items"]:
        base_price *= 1.1  # Slightly higher for perishables
    
    # Add some random variation
    base_price *= np.random.uniform(0.8, 1.2)
    
    base_prices.append(max(base_price, 500))  # Ensure minimum price

df["base_price_inr"] = base_prices

# =========================
# 2. Split Features & Target
# =========================
X = df.drop("base_price_inr", axis=1)
y = df["base_price_inr"]

# =========================
# 3. Define Column Types
# =========================
categorical_features = [
    "pickup_region",
    "drop_region",
    "cargo_type"
]

numeric_features = [
    "distance_km",
    "cargo_weight_kg",
    "cargo_volume_m3",
    "cargo_value_inr"
]

# =========================
# 4. Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# =========================
# 5. XGBoost Regressor
# =========================
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================
# 6. Build Pipeline
# =========================
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

# =========================
# 7. Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. Train Model
# =========================
model_pipeline.fit(X_train, y_train)

# =========================
# 9. Evaluate Model
# =========================
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# =========================
# 10. Prediction Variable (IMPORTANT)
# =========================
def predict_base_price(input_dict):
    """
    input_dict = {
        'pickup_region': 'Tamil Nadu',
        'drop_region': 'Karnataka',
        'distance_km': 350,
        'cargo_weight_kg': 1200,
        'cargo_volume_m3': 6.5,
        'cargo_type': 'Electronics',
        'cargo_value_inr': 500000
    }
    """
    input_df = pd.DataFrame([input_dict])
    predicted_price = model_pipeline.predict(input_df)[0]
    return round(predicted_price, 2)

# =========================
# 11. Example Usage
# =========================
sample_input = {
    "pickup_region": "Tamil Nadu",
    "drop_region": "Karnataka",
    "distance_km": 350,
    "cargo_weight_kg": 1200,
    "cargo_volume_m3": 6.5,
    "cargo_type": "Electronics",
    "cargo_value_inr": 500000
}

base_price_estimate = predict_base_price(sample_input)

print("Predicted Base Price (₹):", base_price_estimate)
