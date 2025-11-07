import re
import math
import string
import unicodedata
import statistics
import emoji
import ftfy
import numpy as np
import pandas as pd
import torch
from collections import Counter
from transformers import AutoTokenizer as _AT, AutoModelForCausalLM
from tqdm.auto import tqdm

from src.config import DEVICE

# --- Perplexity Model Loading ---
# Load models once when module is imported
print("Loading perplexity model (distilgpt2)...")
try:
    _ppl_tok = _AT.from_pretrained("distilgpt2")
    _ppl_mdl = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE).eval()
    print("Perplexity model loaded successfully.")
except Exception as e:
    print(f"Could not load perplexity model: {e}")
    _ppl_tok = None
    _ppl_mdl = None

# --- Feature Helper Functions ---

def normalize_text(s: str) -> str:
    s = ftfy.fix_text(s, normalization="NFKC")
    s = s.replace("\u200b", "")  # zero-width space
    return s

def ascii_ratio(s):
    if not s: return 0.0
    asc = sum(ord(c) < 128 for c in s)
    return asc / len(s)

def emoji_count(s):
    return sum(ch in emoji.EMOJI_DATA for ch in s)

def punct_ratio(s):
    if not s: return 0.0
    punct = sum(1 for c in s if unicodedata.category(c).startswith('P'))
    return punct / len(s)

def digit_ratio(s):
    if not s: return 0.0
    digs = sum(c.isdigit() for c in s)
    return digs / len(s)

def upper_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters: return 0.0
    return sum(c.isupper() for c in letters) / len(letters)

def char_entropy(s):
    if not s: return 0.0
    cnt = Counter(s)
    N = len(s)
    p = np.array([v/N for v in cnt.values()])
    return float(-(p*np.log2(p+1e-12)).sum())

def ttr(s):
    toks = re.findall(r"\w+", s.lower())
    if not toks: return 0.0
    return len(set(toks))/len(toks)

def rep3_rate(s):
    toks = re.findall(r"\w+|\S", s)
    if len(toks) < 3: return 0.0
    grams = Counter(tuple(toks[i:i+3]) for i in range(len(toks)-2))
    tot = max(1, len(toks)-2)
    repeats = sum(v for v in grams.values() if v>1)
    return repeats/tot

@torch.no_grad()
def perplexity(s, max_len=512):
    if not s or not _ppl_mdl: return 100.0
    try:
        enc = _ppl_tok(s, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(DEVICE)
        attn = enc.attention_mask.to(DEVICE)
        out = _ppl_mdl(input_ids, labels=input_ids, attention_mask=attn)
        ppl = torch.exp(out.loss).item()
        return float(min(1e3, ppl))
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        return 100.0

def per_text_features(s):
    s = normalize_text(s)
    return {
        "len_chars": len(s),
        "len_words": len(re.findall(r'\w+', s)),
        "ascii_ratio": ascii_ratio(s),
        "emoji_cnt": emoji_count(s),
        "punct_ratio": punct_ratio(s),
        "digit_ratio": digit_ratio(s),
        "upper_ratio": upper_ratio(s),
        "char_entropy": char_entropy(s),
        "ttr": ttr(s),
        "rep3_rate": rep3_rate(s),
    }

_text_cache = {}
def features_for_pair(a, b, with_ppl=True):
    if a not in _text_cache:
        f = per_text_features(a)
        if with_ppl: f["ppl"] = perplexity(a)
        _text_cache[a] = f
    if b not in _text_cache:
        f = per_text_features(b)
        if with_ppl: f["ppl"] = perplexity(b)
        _text_cache[b] = f
    
    f1, f2 = _text_cache[a], _text_cache[b]
    feats = {}
    for k in f1.keys():
        feats[f"{k}_diff"]  = f1[k] - f2[k]
        feats[f"{k}_ratio"] = (f1[k]+1e-6)/(f2[k]+1e-6)
    return feats

# --- Main Feature Building Function ---

def build_feature_dfs(df_train, df_test, ids_train, ids_test, with_ppl=True):
    """Builds feature DataFrames for train and test sets."""
    print("Building features for training data...")
    pair_feats_train = []
    for i in tqdm(ids_train, desc="Train Features"):
        pair_feats_train.append(features_for_pair(
            df_train.loc[i, "file_1"], df_train.loc[i, "file_2"], with_ppl=with_ppl
        ))
    Xf_train = pd.DataFrame(pair_feats_train, index=ids_train)
    
    print("Building features for test data...")
    pair_feats_test = []
    for i in tqdm(ids_test, desc="Test Features"):
        pair_feats_test.append(features_for_pair(
            df_test.loc[i, "file_1"], df_test.loc[i, "file_2"], with_ppl=with_ppl
        ))
    Xf_test = pd.DataFrame(pair_feats_test, index=ids_test)
    
    print(f"Train features shape: {Xf_train.shape}")
    print(f"Test features shape: {Xf_test.shape}")
    
    return Xf_train, Xf_test