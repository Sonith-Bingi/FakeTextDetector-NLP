import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from typing import Union

from src.config import (
    MODEL_NAME, MAX_LEN, PER_SIDE, DEVICE, SEED, 
    CE_EPOCHS, CE_LR, CE_BATCH_SIZE, CE_FOLDS
)

# --- Tokenizer ---
# Initialize tokenizer once
try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception as e:
    print(f"Failed to load tokenizer for {MODEL_NAME}: {e}")
    exit(1)

# --- Tokenization Function ---
def encode_pair(t1, t2):
    a = TOKENIZER.encode(t1, add_special_tokens=False, truncation=True, max_length=PER_SIDE)
    b = TOKENIZER.encode(t2, add_special_tokens=False, truncation=True, max_length=PER_SIDE)
    input_ids = [TOKENIZER.cls_token_id] + a + [TOKENIZER.sep_token_id] + b + [TOKENIZER.sep_token_id]
    input_ids = input_ids[:MAX_LEN]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn}

# --- Dataset and Collate ---
class PairDataset(Dataset):
    def __init__(self, df_pairs, labels_by_index: Union[pd.Series, None], augment_swap=False):
        self.df_pairs = df_pairs
        self.labels = labels_by_index
        self.augment = augment_swap
        self.rows = []

        for i in df_pairs.index.values:
            t1 = df_pairs.loc[i, "file_1"]
            t2 = df_pairs.loc[i, "file_2"]
            if self.labels is None:
                self.rows.append((i, t1, t2, None, 1))
            else:
                y = int(self.labels.loc[i])  # 1 if file_1 real
                self.rows.append((i, t1, t2, y, 1))
                if augment_swap:
                    self.rows.append((i, t2, t1, 1 - y, -1)) # Swapped pair

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        i, a, b, y, order = self.rows[idx]
        enc = encode_pair(a, b)
        enc["input_ids"] = torch.tensor(enc["input_ids"], dtype=torch.long)
        enc["attention_mask"] = torch.tensor(enc["attention_mask"], dtype=torch.long)
        if y is None:
            return enc, i, order
        return enc, torch.tensor(y, dtype=torch.float), i, order

# --- MOVED THIS CLASS ---
# This class is for _predict_logits, moved to top level to fix pickling error
class _TmpDS(Dataset):
    def __init__(self, df_pairs, swap):
        self.rows = []
        for i in df_pairs.index.values:
            a = df_pairs.loc[i, "file_2"] if swap else df_pairs.loc[i, "file_1"]
            b = df_pairs.loc[i, "file_1"] if swap else df_pairs.loc[i, "file_2"]
            self.rows.append((i, a, b))
    def __len__(self): return len(self.rows)
    def __getitem__(self, k):
        i, a, b = self.rows[k]
        enc = encode_pair(a, b)
        enc["input_ids"] = torch.tensor(enc["input_ids"], dtype=torch.long)
        enc["attention_mask"] = torch.tensor(enc["attention_mask"], dtype=torch.long)
        return enc, i
# --- END MOVED CLASS ---

def collate(batch):
    encs = [b[0] for b in batch]
    maxl = max(len(e["input_ids"]) for e in encs)
    input_ids = torch.zeros((len(batch), maxl), dtype=torch.long)
    attn = torch.zeros((len(batch), maxl), dtype=torch.long)
    ys, ids, orders = [], [], []
    
    for j, (e, *rest) in enumerate(batch):
        li = len(e["input_ids"])
        input_ids[j, :li] = e["input_ids"]
        attn[j, :li] = e["attention_mask"]
        if len(rest) == 3:
            ys.append(rest[0]); ids.append(rest[1]); orders.append(rest[2])
        else:
            ids.append(rest[0]); orders.append(rest[1])
            
    batch_enc = {"input_ids": input_ids, "attention_mask": attn}
    y_tensor = torch.stack(ys) if ys else None
    return batch_enc, y_tensor, ids, orders

# --- Prediction Functions ---
@torch.no_grad()
def _predict_logits(model, df_pairs, batch_size=16, swap=False):
    """Get logits for (t1,t2) or (t2,t1)."""
    
    # --- This class was moved to the top level ---

    def _coll(batch):
        encs = [b[0] for b in batch]; idxs = [b[1] for b in batch]
        maxl = max(len(e["input_ids"]) for e in encs)
        input_ids = torch.zeros((len(batch), maxl), dtype=torch.long)
        attn = torch.zeros((len(batch), maxl), dtype=torch.long)
        for j, e in enumerate(encs):
            li = len(e["input_ids"])
            input_ids[j, :li] = e["input_ids"]
            attn[j, :li] = e["attention_mask"]
        return {"input_ids": input_ids, "attention_mask": attn}, idxs

    ds = _TmpDS(df_pairs, swap=swap)
    # --- CHANGED num_workers=2 to num_workers=0 ---
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_coll, num_workers=0)
    model.eval()
    logits_list = []
    for encs, _ in tqdm(dl, desc="Predicting", leave=False):
        encs = {k: v.to(DEVICE) for k, v in encs.items()}
        logits = model(**encs).logits.squeeze(-1).float().cpu().numpy()
        logits_list.append(logits)
    return np.concatenate(logits_list)

@torch.no_grad()
def predict_pairs_symmetric(model, df_pairs, batch_size=16):
    """Symmetric prediction combining (t1,t2) and (t2,t1)."""
    l12 = _predict_logits(model, df_pairs, batch_size=batch_size, swap=False)
    l21 = _predict_logits(model, df_pairs, batch_size=batch_size, swap=True)
    sym_logit = 0.5 * (l12 - l21)
    return sym_logit

# --- Training Function ---
def train_fold(train_ids, val_ids, df_train, y_series):
    print("Loading model for fold...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CE_LR, weight_decay=0.01)

    df_tr, df_va = df_train.loc[train_ids], df_train.loc[val_ids]
    y_tr, y_va = y_series.loc[train_ids], y_series.loc[val_ids]

    dl_tr = DataLoader(
        PairDataset(df_tr, y_tr, augment_swap=True),
        batch_size=CE_BATCH_SIZE, shuffle=True,
        collate_fn=collate, 
        num_workers=0, # --- CHANGED num_workers=2 to num_workers=0 ---
        pin_memory=True,
    )

    num_steps = CE_EPOCHS * len(dl_tr)
    sched = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_steps), num_steps)
    best_acc, best_state = -1.0, None

    for ep in range(CE_EPOCHS):
        model.train(); losses = []
        for encs, yb, _, _ in tqdm(dl_tr, desc=f"Epoch {ep+1}/{CE_EPOCHS} Training"):
            encs = {k: v.to(DEVICE) for k, v in encs.items()}
            yb = yb.to(DEVICE).unsqueeze(-1)
            
            out = model(**encs)
            loss = F.binary_cross_entropy_with_logits(out.logits, yb)
            
            # Label smoothing / regularization
            loss = (1 - 0.05) * loss + 0.05 * F.binary_cross_entropy_with_logits(
                out.logits, 0.5 * torch.ones_like(yb)
            )
            
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step()
            losses.append(loss.item())

        # Symmetric validation
        model.eval()
        va_logits = predict_pairs_symmetric(model, df_va, batch_size=CE_BATCH_SIZE * 2)
        va_acc = ((va_logits >= 0).astype(int) == y_va.values).mean()
        print(f"  Epoch {ep+1}: loss={np.mean(losses):.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_state)
    return model

# --- Main CV Loop Function ---
def run_ce_training(df_train, df_test, y_series):
    """Runs the full K-Fold cross-validation for the cross-encoder."""
    skf = StratifiedKFold(n_splits=CE_FOLDS, shuffle=True, random_state=SEED)
    oof_logits = np.zeros(len(df_train))
    test_logits_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_train, y_series.values), 1):
        print(f"\n--- Starting Cross-Encoder Fold {fold}/{CE_FOLDS} ---")
        tr_ids, va_ids = df_train.index.values[tr_idx], df_train.index.values[va_idx]
        
        model = train_fold(tr_ids, va_ids, df_train, y_series)
        
        print(f"Validating Fold {fold}...")
        oof_logits[va_idx] = predict_pairs_symmetric(model, df_train.loc[va_ids])
        
        print(f"Predicting on Test Set (Fold {fold})...")
        test_logits_folds.append(predict_pairs_symmetric(model, df_test))

        del model; gc.collect(); torch.cuda.empty_cache()

    oof_acc = ((oof_logits >= 0).astype(int) == y_series.values).mean()
    print(f"\n--- Cross-Encoder Training Complete ---")
    print(f"Cross-encoder OOF accuracy: {oof_acc:.4f}")

    ce_test_logits = np.mean(np.stack(test_logits_folds, axis=0), axis=0)
    return oof_logits, ce_test_logits