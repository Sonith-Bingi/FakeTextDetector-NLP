import numpy as np
import pandas as pd
import warnings
from src.config import set_seed, print_device_info
from src.data_loader import load_data
from src.features import build_feature_dfs
from src.model_ce import run_ce_training
from src.model_blend import run_blender_training

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # 1. Setup
    set_seed()
    print_device_info()
    
    # 2. Load Data
    print("--- Loading Data ---")
    df_train, df_test, y_series, y_values, ids_train, ids_test = load_data()
    
    # 3. Feature Engineering
    print("\n--- Building Features ---")
    Xf_train, Xf_test = build_feature_dfs(df_train, df_test, ids_train, ids_test, with_ppl=True)
    
    # 4. Run Cross-Encoder Model
    print("\n--- Running Cross-Encoder (RoBERTa) ---")
    oof_logits, ce_test_logits = run_ce_training(df_train, df_test, y_series)
    
    # 5. Run Blender Model
    print("\n--- Running Blender (LightGBM) ---")
    blend_test_prob = run_blender_training(
        Xf_train, Xf_test, oof_logits, ce_test_logits, y_values
    )
    
    # 6. Create Submission
    print("\n--- Creating Submission File ---")
    # blend_test_prob is P(file_1 is real)
    # 1 if P >= 0.5, 0 if P < 0.5
    predicted_labels = (blend_test_prob >= 0.5).astype(int)
    
    # Map to submission format: 1 -> 1 (file_1 real), 0 -> 2 (file_2 real)
    submission_real_text_id = np.where(predicted_labels == 1, 1, 2)
    
    submission_df = pd.DataFrame({
        'id': df_test.index,
        'real_text_id': submission_real_text_id
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")
    print("\nFinal submission preview:")
    print(submission_df.head())

if __name__ == "__main__":
    main()