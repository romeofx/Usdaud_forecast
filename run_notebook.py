import papermill as pm
from datetime import datetime
import os

# Define paths
input_notebook = "C:/Users/USER/Desktop/Usdaud_forecast/xauusd_train.ipynb"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = "C:/Users/USER/Desktop/Usdaud_forecast/executed"
output_path = f"{output_folder}/xauusd_train_{timestamp}.ipynb"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Run notebook
pm.execute_notebook(input_notebook, output_path)
