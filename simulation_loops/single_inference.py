import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import evaluate
from scipy.special import softmax

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler


from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

import click

@click.command()
@click.option('--seed',default=565894, help='MonteCarlo Iteration ID.')
@click.option('--dset',default='rel',help='pol/rel/infra dsets')
@click.option('--hf_model', default='conflibert', help="Pretrained model to active train")
def read_opt(seed, dset, hf_model):
    """Read options for training."""
    print(f"{seed=}, {dset=}, {hf_model=}")
    return seed, dset, hf_model

base_path = 'single_res/'

seed, dset, hf_model = read_opt(standalone_mode=False)
#exit(1)
# Hardcoded parameters
#pretrained_model = 'distilbert-base-uncased'

pretrained_model = 'snowood1/ConfliBERT-scr-uncased' #conflibert
if hf_model.lower() == 'bert' :
    pretrained_model = 'bert-base-uncased' #bert
if hf_model.lower() == 'distilbert':
    pretrained_model = 'distilbert-base-uncased' #distilbert

pref=dset.lower().strip()

base_path = base_path.lower().strip().strip('/')
path_rel_data = f'{base_path}/{pref}_full_ds.parquet'
model_path = f'{base_path}/{hf_model}_{pref}_{seed}_model.pt'
print (f'{path_rel_data=} : {model_path=}')

np.random.seed(seed=seed)
torch.manual_seed(seed)

torch.cuda.empty_cache()
mps_device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
accuracy = evaluate.load("accuracy")

def preprocess_function(examples):
    """
    Tokenize text using the chosen Huggingface tokenizer
    The tokenizer is a superglobal.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

def df_to_torch(df, shuffle=True, balance=False):
    """
    Given a Pandas Dataframe containing processed UCDP data,
    return a torch DataLoader that has been processed accordingly to
    :shuffle - randomize the data, for training purposes. Do not use with RNN/LSTM models
    :balance - balance the data to the size of the "1" class
    returns a torch Dataloader with the same data
    """
    if balance:
        df_pos = df[df.label==1]
        df_neg = df[df.label==0].sample(n=df_pos.shape[0], replace=True)
        df = pd.concat([df_pos,df_neg])

    print(Dataset.from_pandas(df))

    r_t = Dataset.from_pandas(df)
    r_t = r_t.map(preprocess_function, batched=True)
    try:
        r_t = r_t.remove_columns(["__index_level_0__"])
        r_t = r_t.remove_columns(["xscore"])
        r_t = r_t.remove_columns(["text"])
    except:
        pass

    try:
        r_t = r_t.rename_column("label", "labels")
    except:
        pass
    r_t.set_format("torch")
    r_t = DataLoader(r_t, shuffle=shuffle, batch_size=8)
    return r_t

def get_predictions(dataloader):
    """
    Given a dataloader and a superglobal trained torch huggingface module
    Since this is inference only, do not gather labels.
    """
    gather_logits = torch.empty(0,device=mps_device)
    gather_labels = torch.empty(0,device=mps_device)

    model.eval()

    progress_bar_x = tqdm(range(int(len(dataloader.dataset)/dataloader.batch_size)))

    for batch in dataloader:
            batch = {k: v.to(mps_device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits

            gather_logits = torch.cat([gather_logits,logits],dim=0)
            gather_labels = torch.cat([gather_labels,batch["labels"]],dim=0)

            predictions = torch.argmax(logits, dim=-1)
            progress_bar_x.update(1)

    probas = softmax(gather_logits.cpu(), axis=1)
    probas = probas[:,1]

    return probas


rel = pd.read_parquet(path_rel_data)

#rel=rel.head(1000)

ucdpid = rel.ucdpid

rel = rel[['article','actual','xscore']].rename(columns={'article':'text','actual':'label'})
rel['text']=rel.text.str.lower()

# These labels are not used, but we need these as int since the model object expects them to be ints
# These are NOT to be used for metric computations of from training
# from here onwards and should be discarded, as they are incorrect.

rel['label'] = rel['label'].fillna(0).astype('int')



print (f"Data loaded with total size: {rel.shape[0]} of which {(rel.xscore>0).sum()} unsupervised 1:s")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = torch.load(model_path)
model.eval()

infer_dataloader = df_to_torch(rel, shuffle=False)
probas = get_predictions(infer_dataloader)

fname = f"{base_path}/preds_full_{hf_model}_{pref}_{seed}_.parquet"
pd.DataFrame({'ucdpid':ucdpid,'preds':probas}).to_parquet(fname)
print("Inference completed successfully!")