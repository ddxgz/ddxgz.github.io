---
title: "Using BERT to perform Topic Tag Prediction for Technical Articles"
description: "Experiments using BERT Mini embeddings and linear SVM for multilabel tag prediction on LinkedInfo articles."
pubDatetime: 2020-02-13T09:20:19Z
modDatetime: 2020-03-31T13:10:14Z
tags:
  [
    "BERT",
    "Machine Learning",
    "Multi-label classification",
    "Text classification",
    "LinkedInfo.co",
  ]
canonicalURL: "https://pcx.linkedinfo.co/post/text-tag-prediction-bert/"
---

## Table of contents

# Introduction

This is a follow up post of [Multi-label classification to predict topic tags of technical articles from LinkedInfo.co](https://pcx.linkedinfo.co/post/text-tag-prediction/). We will continute the same task by using BERT.

Firstly we'll just use the embeddings from BERT, and then feed them to the same classification model used in the last post, SVM with linear kenel. The reason of keep using SVM is that the size of the dataset is quite small.

# Experiments

## Classify by using BERT-Mini and SVM with Linear Kernel

Due to the limited computation capacity, we'll use a smaller BERT model - BERT-Mini. The first experiment we'll try to train on only the titles of the articles.

Now we firstly load the dataset. And then load the pretrained BERT tokenizer and model. Note that we only load the article samples that are in English since the BERT-Mini model here were pretrained in English.

```python
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import nltk
import plotly.express as px
from transformers import (BertPreTrainedModel, AutoTokenizer, AutoModel,
                          BertForSequenceClassification, AdamW, BertModel,
                          BertTokenizer, BertConfig, get_linear_schedule_with_warmup)

import dataset
from mltb.bert import bert_tokenize, bert_transform, get_tokenizer_model, download_once_pretrained_transformers
from mltb.experiment import multilearn_iterative_train_test_split
from mltb.metrics import classification_report_avg


nltk.download('punkt')

RAND_STATE = 20200122
```

```python
ds = dataset.ds_info_tags(from_batch_cache='info', lan='en',
                          concate_title=True,
                          filter_tags_threshold=0, partial_len=3000)
```

```python
c = Counter([tag for tags in ds.target_decoded for tag in tags])

dfc = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(by='count', ascending=False)[:100]

fig_Y = px.bar(dfc, x=dfc.index, y='count',
               text='count',
               labels={'count': 'Number of infos',
                       'x': 'Tags'})
fig_Y.update_traces(texttemplate='%{text}')
```

<figure>
  <iframe
    src="/vega/embed.html?spec=/vega/text-tag-bert-0-spec.json"
    title="Tag frequency chart"
    loading="lazy"
    style="width:100%;border:0;height:460px;"
  ></iframe>
</figure>

```python
dfc_tail = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(by='count', ascending=False)[-200:]

fig_Y = px.bar(dfc_tail, x=dfc_tail.index, y='count',
               text='count',
               labels={'count': 'Number of infos',
                       'x': 'Tags'})
fig_Y.update_traces(texttemplate='%{text}')
```

<figure>
  <iframe
    src="/vega/embed.html?spec=/vega/text-tag-bert-1-spec.json"
    title="Rare tag frequency chart"
    loading="lazy"
    style="width:100%;border:0;height:460px;"
  ></iframe>
</figure>

After we loaded the data, we checked how frequently are the tags being tagged to the articles. Here we only visualized the top-100 tags (you can select area of the figure to zoomin), we can see that there's a big imbalancement of popularity among tags. We can try to mitigate this imbalancement by using different methods like sampling methods and augmentation. But now we'll just pretend we don't know that and leave this aside.

Now let's load the BERT tokenizer and model.

```python
PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
    "google/bert_uncased_L-4_H-256_A-4")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
model = AutoModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)
```

Now we encode all the titles by the BERT-Mini model. We'll use only the 1st output vector from the model as it's used for classification task.

```python
col_text = 'title'
max_length = ds.data[col_text].apply(lambda x: len(nltk.word_tokenize(x))).max()

encoded = ds.data[col_text].apply(
    (lambda x: tokenizer.encode_plus(x, add_special_tokens=True,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     max_length=max_length,
                                     return_tensors='pt')))

input_ids = torch.cat(tuple(encoded.apply(lambda x:x['input_ids'])))
attention_mask = torch.cat(tuple(encoded.apply(lambda x:x['attention_mask'])))

features = []
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0][:, 0, :].numpy()
```

As the features are changed from Tf-idf transformed to BERT transformed, so we'll re-search for the hyper-parameters for the LinearSVC to use.

The scorer we used in grid search is f-0.5 score since we want to weight higher precision over recall.

```python
train_features, test_features, train_labels, test_labels = train_test_split(
    features, ds.target, test_size=0.3, random_state=RAND_STATE)

clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.01, 0.1, 0.5, 1, 10]

parameters = {
    'estimator__penalty': ['l1', 'l2'],
    'estimator__dual': [True, False],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)
print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features)

report = metrics.classification_report(
    test_labels, Y_predicted, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
cols_avg = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
df_report.loc[cols_avg,]
```

    {'estimator__C': 0.1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    0.5793483937857783

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.892857</td>
      <td>0.242326</td>
      <td>0.381194</td>
      <td>1238.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.173746</td>
      <td>0.092542</td>
      <td>0.111124</td>
      <td>1238.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.608618</td>
      <td>0.242326</td>
      <td>0.324186</td>
      <td>1238.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.404088</td>
      <td>0.274188</td>
      <td>0.312305</td>
      <td>1238.0</td>
    </tr>
  </tbody>
</table>
</div>

Though it's not comparable, the result metrics are no better than the Tf-idf one when we use only the English samples with their titles here. The micro average precision is higher, the other averages of precision are about the same. The recalls got much lower.

Now let's combine the titles and short descriptions to see if there's any improvment.

```python
col_text = 'description'
max_length = ds.data[col_text].apply(lambda x: len(nltk.word_tokenize(x))).max()
encoded = ds.data[col_text].apply(
    (lambda x: tokenizer.encode_plus(x, add_special_tokens=True,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     max_length=max_length,
                                     return_tensors='pt')))

input_ids = torch.cat(tuple(encoded.apply(lambda x:x['input_ids'])))
attention_mask = torch.cat(tuple(encoded.apply(lambda x:x['attention_mask'])))

features = []
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0][:, 0, :].numpy()
```

```python
train_features, test_features, train_labels, test_labels = train_test_split(
    features, ds.target, test_size=0.3, random_state=RAND_STATE)
```

```python
clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.1, 1]

parameters = {
    'estimator__penalty': ['l2'],
    'estimator__dual': [True],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features)

classification_report_avg(test_labels, Y_predicted)
```

    {'estimator__C': 1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    0.4954311860243222
                  precision    recall  f1-score  support
    micro avg      0.684015  0.297254  0.414414   1238.0
    macro avg      0.178030  0.109793  0.127622   1238.0
    weighted avg   0.522266  0.297254  0.362237   1238.0
    samples avg    0.401599  0.314649  0.337884   1238.0

There is no improvement, the precision averages even got a little bit worse. Let's try to explore further.

## Iterative stratified multilabel data sampling

It would be a good idea to perform stratified sampling for spliting training and test sets since there's a big imbalancement in the dataset for the labels. The problem is that the size of dataset is very small, which causes it that using normal stratified sampling method would fail since it's likely that some labels may not appear in both training and testing sets. That's why we have to use iterative stratified multilabel sampling. The explanation of this method can refer to [document of scikit-multilearn](http://scikit.ml/stratification.html).

In the code below we have wrapped the split method for brevity.

```python
COL_TEXT = 'description'

train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=0.3, cols=ds.data.columns)

batch_size = 128
model_name = "google/bert_uncased_L-4_H-256_A-4"

train_features, test_features = bert_transform(
    train_features, test_features, COL_TEXT, model_name, batch_size)


clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.1, 1]

parameters = {
    'estimator__penalty': ['l2'],
    'estimator__dual': [True],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features)

print(classification_report_avg(test_labels, Y_predicted))
```

    {'estimator__C': 0.1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    0.3292528001922235
                  precision    recall  f1-score  support
    micro avg      0.674086  0.356003  0.465934   1191.0
    macro avg      0.230836  0.162106  0.181784   1191.0
    weighted avg   0.551619  0.356003  0.420731   1191.0
    samples avg    0.460420  0.377735  0.392599   1191.0

There seems no improvement. But the cross validation F-0.5 score is lower than the testing score. It might be a sign that it's under-fitting.

## Training set augmentation

As the dataset is quite small, now we'll try to augment the trainig set to see if there's any improvement.

Here we set the augmentation level to 2, which means the dataset are concatenated by 2 times of the samples. And the added samples' content will be randomly chopped out as 9/10 of its original content. Of course, both the actions only apply to the training set. The 30% test set is kept aside.

```python
COL_TEXT = 'description'

train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=0.3, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, level=2, crop_ratio=0.1)

batch_size = 128
model_name = "google/bert_uncased_L-4_H-256_A-4"

train_features, test_features = bert_transform(
    train_features, test_features, COL_TEXT, model_name, batch_size)

clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.1, 1]

parameters = {
    'estimator__penalty': ['l2'],
    'estimator__dual': [True],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features)

classification_report_avg(test_labels, Y_predicted)
```

    {'estimator__C': 0.1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    0.9249583214520737
                  precision    recall  f1-score  support
    micro avg      0.616296  0.348409  0.445158   1194.0
    macro avg      0.224752  0.162945  0.180873   1194.0
    weighted avg   0.520024  0.348409  0.406509   1194.0
    samples avg    0.442572  0.373784  0.384738   1194.0

We can see that there's still no improvement. It seems that we should change direction.

## Filter rare tags

If you remember that the first time we loaded the data we visualized the appearence frequency of the tags. It showed that most of the tags appeared only very few times, over 200 tags appeared only once or twice. This is quite a big problem for the model to classify for these tags.

Now let's try to filter out the least appeared tags. Let's start from a big number of 20, i.e., tags appeared in less than 20 articles will be removed.

```python
col_text = 'description'
ds_param = dict(from_batch_cache='info', lan='en',
                concate_title=True,
                filter_tags_threshold=20)
ds = dataset.ds_info_tags(**ds_param)

c = Counter([tag for tags in ds.target_decoded for tag in tags])

dfc = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(by='count', ascending=False)[:100]

fig_Y = px.bar(dfc, x=dfc.index, y='count',
               text='count',
               labels={'count': 'Number of infos',
                       'x': 'Tags'})
fig_Y.update_traces(texttemplate='%{text}')
```

<figure>
  <iframe
    src="/vega/embed.html?spec=/vega/text-tag-bert-2-spec.json"
    title="Top tags zoomed chart"
    loading="lazy"
    style="width:100%;border:0;height:460px;"
  ></iframe>
</figure>

```python
test_size = 0.3
train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=test_size, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, level=2, crop_ratio=0.1)

batch_size = 128
model_name = "google/bert_uncased_L-4_H-256_A-4"

train_features, test_features = bert_transform(
    train_features, test_features, col_text, model_name, batch_size)

clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.1, 1]

parameters = {
    'estimator__penalty': ['l2'],
    'estimator__dual': [True],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)
print(f'Best params in CV: {gs_clf.best_params_}')
print(f'Best score in CV: {gs_clf.best_score_}')

Y_predicted = gs_clf.predict(test_features)

classification_report_avg(test_labels, Y_predicted)
```

    Best params in CV: {'estimator__C': 0.1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    Best score in CV: 0.8943719982878996
                  precision    recall  f1-score  support
    micro avg      0.593583  0.435294  0.502262    765.0
    macro avg      0.523965  0.361293  0.416650    765.0
    weighted avg   0.586632  0.435294  0.490803    765.0
    samples avg    0.458254  0.472063  0.444127    765.0

The filtering of tags made the averages of recall higher, but made the precision lower. The macro average goes up as there're much fewer tags.

## Fine-tuning BERT model

The next step is to see if we can make some progress by fine-tuning the BERT-Mini model. As for a comparable result, the fine-tuning training will be using the same dataset that filtered of tags appear at least in 20 infos. The final classifier model will also be the same of SVM with Linear kernel feeded by the embeddings from the fine-tuned BERT-Mini.

The processing of fine-tuning refers much to [Chris McCormick's post](https://mccormickml.com/2019/07/22/BERT-fine-tuning/).

```python
col_text = 'description'
ds_param = dict(from_batch_cache='info', lan='en',
                concate_title=True,
                filter_tags_threshold=20)
ds = dataset.ds_info_tags(**ds_param)

test_size = 0.3
train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=test_size, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, level=2, crop_ratio=0.1)
```

The `BertForSequenceMultiLabelClassification` class defined below is basically a copy of the `BertForSequenceClassification` class in huggingface's `Transformers`, only with a small change of adding `sigmoid` the logits from classification and adding ` labels = torch.max(labels, 1)[1]` in `forward` for supporting multilabel.

```python
class BertForSequenceMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logtis = torch.sigmoid(logits)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()

                labels = torch.max(labels, 1)[1]

                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
```

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = train_labels.shape[1]
batch_size: int = 16
epochs: int = 4

model_name = download_once_pretrained_transformers(
    "google/bert_uncased_L-4_H-256_A-4")

model = BertForSequenceMultiLabelClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    output_attentions=False,
    output_hidden_states=False,
)

model.to(DEVICE)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], "weight_decay": 0.1,
     },
    {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5,  eps=1e-8  )

tokenizer, model_notuse = get_tokenizer_model(model_name)

input_ids, attention_mask = bert_tokenize(
    tokenizer, train_features, col_text=col_text)
input_ids_test, attention_mask_test = bert_tokenize(
    tokenizer, test_features, col_text=col_text)

train_set = torch.utils.data.TensorDataset(
    input_ids, attention_mask, torch.Tensor(train_labels))
test_set = torch.utils.data.TensorDataset(
    input_ids_test, attention_mask_test, torch.Tensor(test_labels))

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, sampler=RandomSampler(train_set))
test_loader = torch.utils.data.DataLoader(
    test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)


total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)
```

```python
training_stats = []


def best_prec_score(true_labels, predictions):
    fbeta = 0
    thr_bst = 0
    for thr in range(0, 6):
        Y_predicted = (predictions > (thr * 0.1))

        f = metrics.average_precision_score(
            true_labels, Y_predicted, average='micro')
        if f > fbeta:
            fbeta = f
            thr_bst = thr * 0.1

    return fbeta, thr


def train():
    model.train()

    total_train_loss = 0

    for step, (input_ids, masks, labels) in enumerate(train_loader):
        input_ids, masks, labels = input_ids.to(
            DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        model.zero_grad()
        loss, logits = model(input_ids, token_type_ids=None,
                             attention_mask=masks, labels=labels)

        total_train_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print("Train loss: {0:.2f}".format(avg_train_loss))


def val():
    model.eval()

    val_loss = 0

    y_pred, y_true = [], []
    # Evaluate data for one epoch
    for (input_ids, masks, labels) in test_loader:

        input_ids, masks, labels = input_ids.to(
            DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            (loss, logits) = model(input_ids,
                                   token_type_ids=None,
                                   attention_mask=masks,
                                   labels=labels)

        val_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        y_pred += logits.tolist()
        y_true += label_ids.tolist()

    bes_val_prec, bes_val_prec_thr = best_prec_score(
        np.array(y_true), np.array(y_pred))
    y_predicted = (np.array(y_pred) > 0.5)

    avg_val_loss = val_loss / len(test_loader)

    print("Val loss: {0:.2f}".format(avg_val_loss))
    print("best prec: {0:.4f}, thr: {1}".format(
        bes_val_prec, bes_val_prec_thr))
    print(classification_report_avg(y_true, y_predicted))

for ep in range(epochs):
    print(f'-------------- Epoch: {ep+1}/{epochs} --------------')
    train()
    val()

print('-------------- Completed --------------')
```

    -------------- Epoch: 1/4 --------------
    Train loss: 2.94
    Val loss: 2.35
    best prec: 0.1540, thr: 5
                  precision    recall  f1-score  support
    micro avg      0.233079  0.599476  0.335654    764.0
    macro avg      0.185025  0.294674  0.196651    764.0
    weighted avg   0.225475  0.599476  0.292065    764.0
    samples avg    0.252645  0.634959  0.342227    764.0
    -------------- Epoch: 2/4 --------------
    Train loss: 2.14
    Val loss: 1.92
    best prec: 0.1848, thr: 5
                  precision    recall  f1-score  support
    micro avg      0.255676  0.678010  0.371326    764.0
    macro avg      0.381630  0.448064  0.303961    764.0
    weighted avg   0.328057  0.678010  0.355185    764.0
    samples avg    0.273901  0.735660  0.379705    764.0
    -------------- Epoch: 3/4 --------------
    Train loss: 1.78
    Val loss: 1.74
    best prec: 0.1881, thr: 5
                  precision    recall  f1-score  support
    micro avg      0.248974  0.714660  0.369293    764.0
    macro avg      0.272232  0.524172  0.306814    764.0
    weighted avg   0.275572  0.714660  0.364002    764.0
    samples avg    0.273428  0.776291  0.383235    764.0
    -------------- Epoch: 4/4 --------------
    Train loss: 1.61
    Val loss: 1.68
    best prec: 0.1882, thr: 5
                  precision    recall  f1-score  support
    micro avg      0.244105  0.731675  0.366077    764.0
    macro avg      0.288398  0.552318  0.310797    764.0
    weighted avg   0.294521  0.731675  0.369942    764.0
    samples avg    0.267708  0.795730  0.381341    764.0
    -------------- Completed --------------

Save the fine-tuned model for later encoding.

```python
from transformers import WEIGHTS_NAME, CONFIG_NAME, BertTokenizer

output_dir = "./data/models/bert_finetuned_tagthr_20/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)
```

    ('./data/models/bert_finetuned_tagthr_20/vocab.txt',)

Now let's use the fine-tuned model to get the embeddings for the same SVM classification.

```python
batch_size = 128
model_name = output_dir

train_features, test_features = bert_transform(
    train_features, test_features, col_text, model_name, batch_size)


clf = OneVsRestClassifier(LinearSVC())

C_OPTIONS = [0.1, 1, 10]

parameters = {
    'estimator__penalty': ['l2'],
    'estimator__dual': [True],
    'estimator__C': C_OPTIONS,
}

micro_f05_sco = metrics.make_scorer(
    metrics.fbeta_score, beta=0.5, average='micro')

gs_clf = GridSearchCV(clf, parameters,
                      scoring=micro_f05_sco,
                      cv=3, n_jobs=-1)

gs_clf.fit(train_features, train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features)

report = metrics.classification_report(
    test_labels, Y_predicted, output_dict=True)
df_report = pd.DataFrame(report).transpose()
cols_avg = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
df_report.loc[cols_avg]
```

    {'estimator__C': 0.1, 'estimator__dual': True, 'estimator__penalty': 'l2'}
    0.945576388765271

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.793605</td>
      <td>0.714660</td>
      <td>0.752066</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.757259</td>
      <td>0.642333</td>
      <td>0.671768</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.782557</td>
      <td>0.714660</td>
      <td>0.732094</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.798664</td>
      <td>0.762087</td>
      <td>0.754289</td>
      <td>764.0</td>
    </tr>
  </tbody>
</table>
</div>

There's quite a big improvement to both precision and recall after fine-tuning. This result makes the model quite usable.

# Comeback test with tf-idf

Comparing to the early post that the model uses tf-idf to transform the text, we've made some changes to the dataset loading, spliting and augmentation. I'm curious to see if these changes would improve the performance when using tf-idf other than BERT-Mini.

Let's start with samples only in English still.

```python
col_text = 'description'

ds_param = dict(from_batch_cache='info', lan='en',
                concate_title=True,
                filter_tags_threshold=20)
ds = dataset.ds_info_tags(**ds_param)

train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=0.3, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, level=4, crop_ratio=0.2)

clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', dual=True))),
])

C_OPTIONS = [0.1, 1, 10]

parameters = {
    'vect__ngram_range': [(1, 4)],
    'clf__estimator__C': C_OPTIONS,
}
gs_clf = GridSearchCV(clf, parameters, cv=3, n_jobs=-1)
gs_clf.fit(train_features[col_text], train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features[col_text])

classification_report_avg(test_labels, Y_predicted)
```

    {'clf__estimator__C': 10, 'vect__ngram_range': (1, 4)}
    0.9986905637969986

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.955782</td>
      <td>0.367801</td>
      <td>0.531191</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.740347</td>
      <td>0.250587</td>
      <td>0.353632</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.847419</td>
      <td>0.367801</td>
      <td>0.487887</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.452608</td>
      <td>0.396469</td>
      <td>0.412913</td>
      <td>764.0</td>
    </tr>
  </tbody>
</table>
</div>

Now let's try samples in both English and Chinese.

```python
col_text = 'description'

ds_param = dict(from_batch_cache='info', lan=None,
                concate_title=True,
                filter_tags_threshold=20)
ds = dataset.ds_info_tags(**ds_param)

train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=0.3, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, level=4, crop_ratio=0.2)

clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', dual=True))),
])

C_OPTIONS = [0.1, 1, 10]

parameters = {
    'vect__ngram_range': [(1, 4)],
    'clf__estimator__C': C_OPTIONS,
}
gs_clf = GridSearchCV(clf, parameters, cv=3, n_jobs=-1)
gs_clf.fit(train_features[col_text], train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features[col_text])

classification_report_avg(test_labels, Y_predicted)
```

    {'clf__estimator__C': 10, 'vect__ngram_range': (1, 4)}
    0.9962557077625571

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.884273</td>
      <td>0.417952</td>
      <td>0.567619</td>
      <td>1426.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.804614</td>
      <td>0.311396</td>
      <td>0.423867</td>
      <td>1426.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.849494</td>
      <td>0.417952</td>
      <td>0.532041</td>
      <td>1426.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.487522</td>
      <td>0.433421</td>
      <td>0.446447</td>
      <td>1426.0</td>
    </tr>
  </tbody>
</table>
</div>

We can see that, for both the models, the micro average precision is quite high and the recalls are still low. However, the macro averages are much better since we filtered out minority tags.

The model trained on samples with both languages has a lower precisions but higher recalls. The model trained on samples with both languages has a lower precisions but higher recalls. This is reasonable since that in those added Chinese articles, those key terms are more stand out that they can be better captured by tf-idf. That's why the recall goes up a little.

Now lets see how would it perform training on the fulltext.

```python
col_text = 'fulltext'

ds_param = dict(from_batch_cache='fulltext', lan=None,
                concate_title=True,
                filter_tags_threshold=20)
ds = dataset.ds_info_tags(**ds_param)

train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    ds.data, ds.target, test_size=0.3, cols=ds.data.columns)

train_features, train_labels = dataset.augmented_samples(
    train_features, train_labels, col=col_text, level=4, crop_ratio=0.2)

clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', dual=True))),
])

C_OPTIONS = [0.1, 1, 10]

parameters = {
    'vect__ngram_range': [(1, 4)],
    'clf__estimator__C': C_OPTIONS,
}
gs_clf = GridSearchCV(clf, parameters, cv=3, n_jobs=-1)
gs_clf.fit(train_features[col_text], train_labels)

print(gs_clf.best_params_)
print(gs_clf.best_score_)

Y_predicted = gs_clf.predict(test_features[col_text])

classification_report_avg(test_labels, Y_predicted)
```

```python
{'clf__estimator__C': 10, 'vect__ngram_range': (1, 4)}
0.9719756244169426
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.891927</td>
      <td>0.479692</td>
      <td>0.623862</td>
      <td>1428.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.858733</td>
      <td>0.385982</td>
      <td>0.502745</td>
      <td>1428.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.864123</td>
      <td>0.479692</td>
      <td>0.584996</td>
      <td>1428.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.559050</td>
      <td>0.494357</td>
      <td>0.510951</td>
      <td>1428.0</td>
    </tr>
  </tbody>
</table>
</div>

The model trained on the fulltext is slightly better, but the training time is way much longer. A tuning on the length of the partial text could be explored with both tf-idf and BERT.

# Final thoughts

Comparing to the early naive model described in the previous post, we had much improment by stratified sampling, simple training set augmentation, and especially filter out rare tags.

When comparing the using of embedding/encoding between BERT-Mini and tf-idf, one had better recall and the other had better precision, which is reasonable. tf-idf is mainly capturing those key terms, but not able to understand the semantic meaning of the text. That's why it had high precision if certain keywords are mentioned in the text, but low recall if when some tags are hiding behind the semantic mearning of the text. While BERT is powerful to capture some semantic meaning of the text, that leads to higher recall.

Given that we have only a very small dataset used for the training, we certainly can have further improvement by using external dataset (the questions and answers from Stack Overflow is an ideal source) and adding additional tokens to BERT's vocabulary (technica articles is somehow a slightly different domain area that has dfferent terms and language usage comparing).
