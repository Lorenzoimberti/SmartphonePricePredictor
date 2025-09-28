from src.data_loader import load_data
from src.evaluation import evaluate_model
from src.utils import generate_random_smartphone, convert_inr_to_usd
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

zip_path = "data/raw/smartphoneDataset.zip"  # Path to zip file
extract_to = "data/raw/"  # Extraction folder

with zipfile.ZipFile(zip_path, "r") as zip_ref:  # Extract ZIP
    zip_ref.extractall(extract_to)

df = load_data(extract_to + "ndtv_data_final.csv")  # Load CSV
print("Dataset loaded")

target_col = "Price"  # Target column

X = df.drop(target_col, axis=1)  # Features
y = df[target_col]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/test split
print("Dataset prepared")

categorical_features = ["Brand", "Touchscreen", "Operating system", "Wi-Fi", "Bluetooth", "GPS", "3G", "4G/ LTE", "Model", "Name"]  # Categorical columns
numeric_features = [col for col in X_train.columns if col not in categorical_features]  # Numeric columns

preprocessor = ColumnTransformer(  # Preprocessing pipeline
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])  # Full pipeline

pipeline.fit(X_train, y_train)  # Train model
print("Model trained")

metrics = evaluate_model(pipeline, X_test, y_test)  # Evaluate model
print("Model evaluated. Performance metrics:", metrics)

new_phone = generate_random_smartphone()  # Generate random smartphone
new_phone_df = pd.DataFrame([new_phone])  # Convert dict to DataFrame

new_phone_dict = new_phone_df.iloc[0].to_dict()
for key, value in new_phone_dict.items():
    print(f"{key} = {value}")

price_pred = pipeline.predict(new_phone_df)[0]  # Predict price
price_usd = convert_inr_to_usd(price_pred)
print(f"Predicted price: {price_usd:.2f}")
