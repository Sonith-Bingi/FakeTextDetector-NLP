import os
import re
import zipfile
import pandas as pd
from src.config import ZIP_FILE_PATH, EXTRACT_DIR, KAGGLE_TRAIN_DIR, KAGGLE_TEST_DIR, KAGGLE_TRAIN_CSV

def unzip_data():
    """Unzips the data.zip file."""
    if os.path.exists(EXTRACT_DIR):
        print(f"Data directory '{EXTRACT_DIR}' already exists. Skipping unzipping.")
        return
    
    print(f"Extracting '{ZIP_FILE_PATH}' to '{EXTRACT_DIR}'...")
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Data extracted successfully.")
    except FileNotFoundError:
        print(f"Error: '{ZIP_FILE_PATH}' not found.")
        print("Please make sure 'data.zip' is in the root project directory.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        exit(1)

def read_texts_from_dir(dir_path):
    """Reads file_1.txt and file_2.txt from subdirectories."""
    data = []
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found at {dir_path}")
        return pd.DataFrame(columns=["id", "file_1", "file_2"]).set_index("id")
        
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, "file_1.txt"), "r", encoding="utf-8") as f1:
                    t1 = f1.read().strip()
                with open(os.path.join(folder_path, "file_2.txt"), "r", encoding="utf-8") as f2:
                    t2 = f2.read().strip()
                
                m = re.findall(r"(\d+)$", folder_name)
                idx = int(m[0]) if m else len(data)
                data.append((idx, t1, t2))
            except Exception as e:
                print(f"WARN: Could not read files in {folder_name}. Error: {e}")
                
    df = pd.DataFrame(data, columns=["id", "file_1", "file_2"]).set_index("id").sort_index()
    return df

def load_data():
    """Main function to unzip and load all datasets."""
    unzip_data()
    
    print("Loading datasets...")
    df_train = read_texts_from_dir(KAGGLE_TRAIN_DIR)
    df_test = read_texts_from_dir(KAGGLE_TEST_DIR)
    df_train_gt = pd.read_csv(KAGGLE_TRAIN_CSV).set_index("id").sort_index()
    
    # y=1 if file_1 is REAL, else 0
    y_series = (df_train_gt["real_text_id"] == 1).astype(int)
    y_values = y_series.values
    
    ids_train = df_train.index.values
    ids_test = df_test.index.values
    
    print(f"Train pairs loaded: {len(df_train)}")
    print(f"Test pairs loaded: {len(df_test)}")
    print(f"Train labels loaded: {len(y_series)}")
    print(f"Label distribution (1=file_1 real): \n{y_series.value_counts()}")
    
    return df_train, df_test, y_series, y_values, ids_train, ids_test