import random
import pandas as pd
import requests

def generate_random_smartphone():
    brands = ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Oppo"]
    operating_systems = ["Android", "iOS"]
    yes_no = ["Yes", "No"]

    smartphone = {
        #"ID": random.randint(1000, 9999),
        "Name": f"Phone{random.randint(100, 999)}",
        "Brand": random.choice(brands),
        "Model": f"M{random.randint(1, 50)}",
        "Battery capacity (mAh)": random.randint(2000, 6000),
        "Screen size (inches)": round(random.uniform(4.5, 7.0), 2),
        "Touchscreen": random.choice(yes_no),
        "Resolution x": random.randint(720, 1440),
        "Resolution y": random.randint(1280, 3200),
        "Processor": random.randint(1, 16),  # core numbers
        "RAM (MB)": random.choice([2048, 3072, 4096, 6144, 8192]),
        "Internal storage (GB)": random.choice([32, 64, 128, 256, 512]),
        "Rear camera": round(random.uniform(8, 108), 1),  # megapixel
        "Front camera": round(random.uniform(5, 32), 1),
        "Operating system": random.choice(operating_systems),
        "Wi-Fi": random.choice(yes_no),
        "Bluetooth": random.choice(yes_no),
        "GPS": random.choice(yes_no),
        "Number of SIMs": random.choice([1, 2]),
        "3G": random.choice(yes_no),
        "4G/ LTE": random.choice(yes_no),
        #"Price": random.randint(5000, 150000)  # INR
    }

    return smartphone


def transform_new_phone(new_phone, X_train_columns):
    # From dictionary to DataFrame 2D
    df = pd.DataFrame([new_phone])

    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=False)

    # Missing columns with value 0
    missing_cols = set(X_train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Order columns
    df = df[X_train_columns]

    return df


import requests

def convert_inr_to_usd(amount_inr):
    url = f"https://api.frankfurter.app/latest?amount={amount_inr}&from=INR&to=USD"

    try:
        response = requests.get(url)
        data = response.json()

        if "rates" in data and "USD" in data["rates"]:
            return data["rates"]["USD"]
        else:
            print("Error: unexpected response", data)
            return None
    except Exception as e:
        print("Currency conversion error:", e)
        return None

