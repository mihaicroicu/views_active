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
@click.option('--seed',default=42, help='MonteCarlo Iteration ID.')
@click.option('--active_learner',default=100, help='Active Learner Batch size')
@click.option('--adam_lr',default=5e-5, help='Adam LR')
@click.option('--rounds',default=7, help='Active Learning Rounds')
@click.option('--rel',default=True, help='Religion or Politics')
def read_opt(seed, active_learner, adam_lr, rounds, rel):
    """Read options for training."""
    print(f"{seed=}, {active_learner=}, {adam_lr=}, {rounds=}, {rel=}")
    return seed, active_learner, adam_lr, rounds, rel

seed, active_learner, adam_lr, rounds, rel = read_opt(standalone_mode=False)

# Hardcoded parameters
#pretrained_model = 'distilbert-base-uncased'
pretrained_model = 'snowood1/ConfliBERT-scr-uncased'

if rel:
    path_rel_train = '/mimer/NOBACKUP/groups/snic2019-3-404/text_jocke_marisol/rel_fixed_train.parquet'
    path_rel_test = '/mimer/NOBACKUP/groups/snic2019-3-404/text_jocke_marisol/rel_fixed_test.parquet'
    d_prefix = 'rel'
else:
    path_rel_train = '/mimer/NOBACKUP/groups/snic2019-3-404/text_jocke_marisol/pol_fixed_train.parquet'
    path_rel_test = '/mimer/NOBACKUP/groups/snic2019-3-404/text_jocke_marisol/pol_fixed_test.parquet'
    d_prefix = 'pol'

fraction_in_train = .8
active_batch_size =  active_learner
seed = seed
adam_learning_rate = float(adam_lr) #5e-5
rounds = rounds

# Initialization
np.random.seed(seed=seed)
torch.manual_seed(seed)

# Clean up any weights stored on GPU and make sure retraining doesn't happen on top 
# Of already trained data.

try:
    del model
except:
    pass

try:
    del trainer
except:
    pass

torch.cuda.empty_cache()
mps_device = torch.device("cuda")


# Helper functions for the script

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

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
    r_t = DataLoader(r_t, shuffle=shuffle, batch_size=16)
    return r_t



def roc_pr_scorer(labels, probas):
    """
    Computes and returns AUROC and AUPR (avg. prec.) scores given actual labels and scores.
    """
    aupr_wv = average_precision_score(labels, probas)
    auroc_wv = roc_auc_score(labels, probas)
    return aupr_wv, auroc_wv


def get_predictions(dataloader):
    """
    Given a dataloader and a superglobal trained torch huggingface module
    Predicts logits from
    """
    gather_logits = torch.empty(0,device=mps_device)
    gather_labels = torch.empty(0,device=mps_device)

    metric = evaluate.load("accuracy")
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
            metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_x.update(1)
            
    probas = softmax(gather_logits.cpu(), axis=1)
    probas = probas[:,1]
    labels = np.array(gather_labels.cpu())
    
    return labels, probas, metric.compute()

# Load data and preprocess it
rel_train = pd.read_parquet(path_rel_train)
rel_train = rel_train[['article','actual','xscore']].rename(columns={'article':'text','actual':'label'})
rel_train['text']=rel_train.text.str.lower()
print (f"Data loaded with total size: {rel_train.shape[0]} of which {(rel_train.xscore>0).sum()} unsupervised 1:s")

rel_test = pd.read_parquet(path_rel_test)
rel_test = rel_test[['article','actual','xscore']].rename(columns={'article':'text','actual':'label'})
rel_test['text']=rel_test.text.str.lower()
print (f"Data loaded with total size: {rel_test.shape[0]} of which {(rel_test.xscore>0).sum()} unsupervised 1:s")


r_train = rel_train.sample(frac=fraction_in_train)
r_test = rel_test

# 
# This is the function sampling the unsupervised dictionary/POS analysis for preliminary active learning
# And creates the seed dataset that is then sent to the human coder for initial labelling
# 
# Since this is simulating the process (the whole dataset is human-labeled for experimentation)
# No need to hook a coding platform here.

prob_pos = r_train.sample(n=active_batch_size, weights=rel_train.xscore**4)
prob_neg = r_train[r_train.xscore==0].sample(n=active_batch_size)
active_seed = pd.concat([prob_pos,prob_neg])

# Make the training seed, the eval full dataset into a torch dataset
# Also make a tiny 200 balanced dataset for small experimental work if needed

r_tiny = pd.concat([r_test[r_test.label==1].sample(100),r_test[r_test.label==0].sample(100)])
tiny_dataloader = df_to_torch(r_tiny, shuffle=False)

eval_dataloader = df_to_torch(r_test, shuffle=False)
train_active_seed = df_to_torch(active_seed)

# Import the Huggingface model and associated pretrained weights

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model, num_labels=2, id2label=id2label, label2id=label2id
)

# Set up the experiment for re-training and send to GPU

optimizer = AdamW(model.parameters(), lr=adam_learning_rate)


num_epochs = rounds #20
num_training_steps = num_epochs * (int(len(train_active_seed.dataset)/8)+1)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(mps_device)

train_dataloader = train_active_seed

# Fetch the weights and start the active training loop
model.train()

# Each epoch corresponds to a new active learning injection, not just a single training pass!
for epoch in range(num_epochs):
    
    print (f"Epoch {epoch}:")
    progress_bar = tqdm(range(int(train_dataloader.dataset.num_rows/8)))
    
    # At each epoch:
    # Predict the held-back eval dataset with the model in its current training state
    # And give me metrics (AUPR(AP)/AUROC and acc). Save them to disk
    # These are used to monitor the experiment's speed.
    # Print out the current metrics against the held-back evaluation dataset
    labels, probas, acc = get_predictions(eval_dataloader)
    ap, auroc = roc_pr_scorer(labels, probas)
    print (f"{epoch=} :: {ap=}, {auroc=}, {acc=}")
    pd.DataFrame({'seed':[seed],'epoch':[epoch],'lr':[adam_learning_rate],'bs':[active_batch_size],
             'ap':[ap], 'auroc':[auroc], 'acc':[acc['accuracy']]}).to_csv(f'{d_prefix}_fixed/agg.csv',mode='a+')
    
    # In each epoch, fetch a new shard of train+test from the pre-prepared shard list.
    #train_dataloader = clean_tokenized_trainer(tokenized_train_shards[epoch])
    
    # Train the model by updating the weights with a full pass over the 
    for batch in train_dataloader:
        batch = {k: v.to(mps_device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    likely_cands = pd.concat([r_train[r_train.xscore>=1],r_train[r_train.xscore==0].sample(n=1000)])
    likely_cands_torch = df_to_torch(likely_cands, False)
    likely_cands_preds = get_predictions(likely_cands_torch)
    likely_cands['xscore'] = likely_cands_preds[1]
    new_iter = likely_cands[likely_cands.xscore.between(0.5,1)].sample(n=active_batch_size)
    active_seed = pd.concat([new_iter, active_seed])
    print (f"At {epoch=} dataset size is {active_seed.shape[0]}")
    train_dataloader = df_to_torch(active_seed, balance=True)

    
labels, probas, acc = get_predictions(eval_dataloader)

torch.save(model.state_dict(), f'{d_prefix}_fixed/{d_prefix}_{seed}_tensors.pt')
fname = f"{d_prefix}_fixed/experiment_{seed}_batch_{active_batch_size}_lr_{adam_learning_rate}.parquet"
pd.DataFrame({'actuals':labels,'preds':probas.numpy()}).to_parquet(fname)
