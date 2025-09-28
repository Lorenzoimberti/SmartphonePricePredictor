import pandas as pd

def predict(model, smartphone_specs):
    # MUST be a DataFrame
    return model.predict(smartphone_specs)[0]