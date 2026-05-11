import pandas as pd
from datasets import load_dataset

def load_xnli_subset(split="validation", num_samples=1000):
    """Loads a subset of English and Turkish XNLI data for testing."""
    dataset_en = load_dataset("xnli", "en", split=f"{split}[:{num_samples}]")
    dataset_tr = load_dataset("xnli", "tr", split=f"{split}[:{num_samples}]")
    
    df_en = pd.DataFrame(dataset_en)
    df_tr = pd.DataFrame(dataset_tr)
    df = pd.concat([df_en, df_tr]).reset_index(drop=True)
    
    # Combine premise and hypothesis for embedding
    df['text'] = df['premise'] + " [SEP] " + df['hypothesis']
    return df
