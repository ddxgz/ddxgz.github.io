---
title: "A Walk Through of the IEEE-CIS Fraud Detection Challenge"
description: "Walkthrough of the IEEE-CIS fraud detection challenge with feature analysis and model experiments."
pubDatetime: 2020-02-10T09:07:11Z
modDatetime: 2020-02-10T09:07:11Z
tags: ["Machine Learning", "Fraud Detection"]
canonicalURL: "https://pcx.linkedinfo.co/post/fraud-detection/"
---

# Introduction

This is a brief walk through of the Kaggle challenge IEEE-CIS Fraud Detection. The process in this post is not meant to compete the top solution by performing an extre feature engineering and a greedy search for the best model with hyper-parameters. This is just to walk through the problem and demonstrate a relatively good solution, by doing feature analysis and a few experiments with reference to other's methods.

The problem of this challenge is to detect payment frauds by using the data of the transactions and identities. The performance of the prediction is evaluated on _ROC AUC_. The reason why this measure is suitable for this problem (rather than Precision-Recall) can refer to the discussion [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/99982).

# Look into the data

The provided dataset is broken into two files named `identity` and `transaction`, which are joined by `TransactionID` (note that NOT all the transactions have corresponding identity information).

### Transaction Table

- TransactionDT: timedelta from a given reference datetime (not an actual
  timestamp), the number of seconds in a day (60 _ 60 _ 24 = 86400)
- TransactionAMT: transaction payment amount in USD
- ProductCD: product code, the product for each transaction
- card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
- addr: address
- dist: distance
- P\_ and (R\_\_) emaildomain: purchaser and recipient email domain
- C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
- D1-D15: timedelta, such as days between previous transaction, etc.
- M1-M9: match, such as names on card and address, etc.
- Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

Among these variables, categorical variables are:

- ProductCD
- card1 - card6
- addr1, addr2
- Pemaildomain Remaildomain
- M1 - M9

### Identity Table

All the variable in this table are categorical:

- DeviceType
- DeviceInfo
- id12 - id38

A more detailed explanation of the data can be found in the reply of [this discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203).

Now let's have a close look at the data.

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

```python
import numpy as np
import pandas as pd
import plotly.express as px


DATA_DIR = '/content/drive/My Drive/colab-data/fraud detect/data'

tran_train = reduce_mem_usage(pd.read_csv(f'{DATA_DIR}/train_transaction.csv'))
id_train = reduce_mem_usage(pd.read_csv(f'{DATA_DIR}/train_identity.csv'))

tran_train.info()
tran_train.head()
id_train.info()
id_train.head()
```

    Mem. usage decreased to 542.35 Mb (69.4% reduction)
    Mem. usage decreased to 25.86 Mb (42.7% reduction)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 590540 entries, 0 to 590539
    Columns: 394 entries, TransactionID to V339
    dtypes: float16(332), float32(44), int16(1), int32(2), int8(1), object(14)
    memory usage: 542.3+ MB

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>...</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>V322</th>
      <th>V323</th>
      <th>V324</th>
      <th>V325</th>
      <th>V326</th>
      <th>V327</th>
      <th>V328</th>
      <th>V329</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.5</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.0</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.0</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>outlook.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2987003</td>
      <td>0</td>
      <td>86499</td>
      <td>50.0</td>
      <td>W</td>
      <td>18132</td>
      <td>567.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>debit</td>
      <td>476.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yahoo.com</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>112.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>1758.0</td>
      <td>925.0</td>
      <td>0.0</td>
      <td>354.0</td>
      <td>0.0</td>
      <td>135.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>1404.0</td>
      <td>790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2987004</td>
      <td>0</td>
      <td>86506</td>
      <td>50.0</td>
      <td>H</td>
      <td>4497</td>
      <td>514.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>420.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 394 columns</p>
</div>

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 144233 entries, 0 to 144232
    Data columns (total 41 columns):
    TransactionID    144233 non-null int32
    id_01            144233 non-null float16
    id_02            140872 non-null float32
    id_03            66324 non-null float16
    id_04            66324 non-null float16
    id_05            136865 non-null float16
    id_06            136865 non-null float16
    id_07            5155 non-null float16
    id_08            5155 non-null float16
    id_09            74926 non-null float16
    id_10            74926 non-null float16
    id_11            140978 non-null float16
    id_12            144233 non-null object
    id_13            127320 non-null float16
    id_14            80044 non-null float16
    id_15            140985 non-null object
    id_16            129340 non-null object
    id_17            139369 non-null float16
    id_18            45113 non-null float16
    id_19            139318 non-null float16
    id_20            139261 non-null float16
    id_21            5159 non-null float16
    id_22            5169 non-null float16
    id_23            5169 non-null object
    id_24            4747 non-null float16
    id_25            5132 non-null float16
    id_26            5163 non-null float16
    id_27            5169 non-null object
    id_28            140978 non-null object
    id_29            140978 non-null object
    id_30            77565 non-null object
    id_31            140282 non-null object
    id_32            77586 non-null float16
    id_33            73289 non-null object
    id_34            77805 non-null object
    id_35            140985 non-null object
    id_36            140985 non-null object
    id_37            140985 non-null object
    id_38            140985 non-null object
    DeviceType       140810 non-null object
    DeviceInfo       118666 non-null object
    dtypes: float16(22), float32(1), int32(1), object(17)
    memory usage: 25.9+ MB

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_13</th>
      <th>id_14</th>
      <th>id_15</th>
      <th>id_16</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_23</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_27</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_30</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2987004</td>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-480.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>Android 7.0</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2987008</td>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>49.0</td>
      <td>-300.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>621.0</td>
      <td>500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>iOS 11.1.2</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2987010</td>
      <td>-5.0</td>
      <td>191631.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>410.0</td>
      <td>142.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>Windows</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2987011</td>
      <td>-5.0</td>
      <td>221832.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>225.0</td>
      <td>NaN</td>
      <td>176.0</td>
      <td>507.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2987016</td>
      <td>0.0</td>
      <td>7460.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-300.0</td>
      <td>Found</td>
      <td>Found</td>
      <td>166.0</td>
      <td>15.0</td>
      <td>529.0</td>
      <td>575.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>Mac OS X 10_11_6</td>
      <td>chrome 62.0</td>
      <td>24.0</td>
      <td>1280x800</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>MacOS</td>
    </tr>
  </tbody>
</table>
</div>

```python
is_fraud = tran_train[['isFraud', 'TransactionID']].groupby('isFraud').count()

is_fraud['ratio'] = is_fraud['TransactionID'] / is_fraud['TransactionID'].sum()
fig_Y = px.bar(is_fraud, x=is_fraud.index, y='TransactionID',
               text='ratio',
               labels={'TransactionID': 'Number of transactions',
                       'x': 'is fraud'})
fig_Y.update_traces(texttemplate='%{text:.6p}')
```

<figure>
  <iframe
    src="/vega/embed.html?spec=/vega/fraud-detection-0-spec.json"
    title="Fraud vs non-fraud distribution"
    loading="lazy"
    style="width:100%;border:0;min-height:360px;"
  ></iframe>
</figure>

## Very imbalanced target varible

Positives of `isFraud` is very low of 3.5% in the entire dataset. For this classification problem, it's very important to have high true positive rate. That is, how good can the model identify the fraud cases from all the fraud cases. So recall is in a sense more important than precision in this problem. Macro average of recall would be a good side metric for this problem. Of cource, in reality we need to consider the belance between the cost of a few frauds and the cost of handling cases.

In addition, we need to put some effort on the sampling and train-val split method, to ensure that the minority class samples have enough impact to the model while training. Class weights of the model could be set to see if there's difference in performance.

## Check missing values

Now let's have a look at if there's any missing value in the dataset. We can see from the table below that there're quite a lot of missing values.

It's hard to tell how we should handle with them before we look into each variable. Sometimes a missing value stands for something. It also depends on what kind of model we are going to use. We can leave them as missing value when using a tree model.

```python
def missing_ratio_col(df):
    df_na = (df.isna().sum() / len(df)) * 100
    if isinstance(df, pd.DataFrame):
        df_na = df_na.drop(
            df_na[df_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame(
            {'Missing Ratio %': df_na})
    else:
        missing_data = pd.DataFrame(
            {'Missing Ratio %': df_na}, index=[0])

    return missing_data

missing_ratio_col(tran_train)
missing_ratio_col(id_train)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Ratio %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dist2</th>
      <td>93.628374</td>
    </tr>
    <tr>
      <th>D7</th>
      <td>93.409930</td>
    </tr>
    <tr>
      <th>D13</th>
      <td>89.509263</td>
    </tr>
    <tr>
      <th>D14</th>
      <td>89.469469</td>
    </tr>
    <tr>
      <th>D12</th>
      <td>89.041047</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>V307</th>
      <td>0.002032</td>
    </tr>
    <tr>
      <th>V299</th>
      <td>0.002032</td>
    </tr>
    <tr>
      <th>V309</th>
      <td>0.002032</td>
    </tr>
    <tr>
      <th>V310</th>
      <td>0.002032</td>
    </tr>
    <tr>
      <th>V308</th>
      <td>0.002032</td>
    </tr>
  </tbody>
</table>
<p>374 rows × 1 columns</p>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Ratio %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id_24</th>
      <td>96.708798</td>
    </tr>
    <tr>
      <th>id_25</th>
      <td>96.441868</td>
    </tr>
    <tr>
      <th>id_07</th>
      <td>96.425922</td>
    </tr>
    <tr>
      <th>id_08</th>
      <td>96.425922</td>
    </tr>
    <tr>
      <th>id_21</th>
      <td>96.423149</td>
    </tr>
    <tr>
      <th>id_26</th>
      <td>96.420375</td>
    </tr>
    <tr>
      <th>id_27</th>
      <td>96.416215</td>
    </tr>
    <tr>
      <th>id_23</th>
      <td>96.416215</td>
    </tr>
    <tr>
      <th>id_22</th>
      <td>96.416215</td>
    </tr>
    <tr>
      <th>id_18</th>
      <td>68.722137</td>
    </tr>
    <tr>
      <th>id_04</th>
      <td>54.016071</td>
    </tr>
    <tr>
      <th>id_03</th>
      <td>54.016071</td>
    </tr>
    <tr>
      <th>id_33</th>
      <td>49.187079</td>
    </tr>
    <tr>
      <th>id_10</th>
      <td>48.052110</td>
    </tr>
    <tr>
      <th>id_09</th>
      <td>48.052110</td>
    </tr>
    <tr>
      <th>id_30</th>
      <td>46.222432</td>
    </tr>
    <tr>
      <th>id_32</th>
      <td>46.207872</td>
    </tr>
    <tr>
      <th>id_34</th>
      <td>46.056034</td>
    </tr>
    <tr>
      <th>id_14</th>
      <td>44.503685</td>
    </tr>
    <tr>
      <th>DeviceInfo</th>
      <td>17.726179</td>
    </tr>
    <tr>
      <th>id_13</th>
      <td>11.726165</td>
    </tr>
    <tr>
      <th>id_16</th>
      <td>10.325654</td>
    </tr>
    <tr>
      <th>id_05</th>
      <td>5.108401</td>
    </tr>
    <tr>
      <th>id_06</th>
      <td>5.108401</td>
    </tr>
    <tr>
      <th>id_20</th>
      <td>3.447200</td>
    </tr>
    <tr>
      <th>id_19</th>
      <td>3.407681</td>
    </tr>
    <tr>
      <th>id_17</th>
      <td>3.372321</td>
    </tr>
    <tr>
      <th>id_31</th>
      <td>2.739318</td>
    </tr>
    <tr>
      <th>DeviceType</th>
      <td>2.373243</td>
    </tr>
    <tr>
      <th>id_02</th>
      <td>2.330257</td>
    </tr>
    <tr>
      <th>id_11</th>
      <td>2.256765</td>
    </tr>
    <tr>
      <th>id_28</th>
      <td>2.256765</td>
    </tr>
    <tr>
      <th>id_29</th>
      <td>2.256765</td>
    </tr>
    <tr>
      <th>id_35</th>
      <td>2.251912</td>
    </tr>
    <tr>
      <th>id_36</th>
      <td>2.251912</td>
    </tr>
    <tr>
      <th>id_15</th>
      <td>2.251912</td>
    </tr>
    <tr>
      <th>id_37</th>
      <td>2.251912</td>
    </tr>
    <tr>
      <th>id_38</th>
      <td>2.251912</td>
    </tr>
  </tbody>
</table>
</div>

## Detailed look at each variable

There're very good references of EDA and feature engineering on the dataset, so it's meaningless to repeat here. Please check the list here if you're interested:

- [Feature Engineering Techniques](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575)
- [EDA for Columns V and ID](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id#EDA-for-Columns-V-and-ID)
- [XGB Fraud with Magic scores LB 0.96](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600#Feature-Engineering)

# Data transformation pipeline

Based on the references and my own analysis, here we have a pipeline of the transformations to perform on the dataset. It can be adjusted for experimenting. Explanation of the transformations see in code comments.

```python
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from typing import List, Callable


DATA_DIR = '/content/drive/My Drive/colab-data/fraud detect/data'


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_df(test_set: bool = False, nrows: int = None, sample_ratio: float = None, reduce_mem: bool = True) -> pd.DataFrame:
    if test_set:
        tran = pd.read_csv(f'{DATA_DIR}/test_transaction.csv', nrows=nrows)
        ids = pd.read_csv(f'{DATA_DIR}/test_identity.csv', nrows=nrows)
    else:
        tran = pd.read_csv(f'{DATA_DIR}/train_transaction.csv', nrows=nrows)
        ids = pd.read_csv(f'{DATA_DIR}/train_identity.csv', nrows=nrows)

    if sample_ratio:
        size = int(len(tran) * sample_ratio)
        tran = tran.sample(n=size, random_state=RAND_STATE)
        ids = ids.sample(n=size, random_state=RAND_STATE)
    df = tran.merge(ids, how='left', on='TransactionID')
    if reduce_mem:
        reduce_mem_usage(df)
    return df


def cat_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    cols.append('ProductCD')

    cols_card = [c for c in df.columns if 'card' in c]
    cols.extend(cols_card)

    cols_addr = ['addr1', 'addr2']
    cols.extend(cols_addr)

    cols_emaildomain = [c for c in df if 'email' in c]
    cols.extend(cols_emaildomain)

    cols_M = [c for c in df if c.startswith('M')]
    cols.extend(cols_M)

    cols.extend(['DeviceType', 'DeviceInfo'])

    cols_id = [c for c in df if c.startswith('id')]
    cols.extend(cols_id)

    return cols


def num_cols(df: pd.DataFrame, target_col='isFraud') -> List[str]:
    cols_cat = cat_cols(df)
    cats = df[cols_cat]
    cols_num = list(set(df.columns) - set(cols_cat))

    if target_col in cols_num:
        cols_num.remove(target_col)

    return cols_num


def missing_ratio_col(df):
    df_na = (df.isna().sum() / len(df)) * 100
    if isinstance(df, pd.DataFrame):
        df_na = df_na.drop(
            df_na[df_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %': df_na})
    else:
        missing_data = pd.DataFrame({'Missing Ratio %': df_na}, index=[0])

    return missing_data


class NumColsNaMedianFiller(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)
        cols_num = list(set(df.columns) - set(cols_cat))

        for col in cols_num:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

        return df


class NumColsNegFiller(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_num = num_cols(df)

        for col in cols_num:
            df[col].fillna(-999, inplace=True)

        return df


class NumColsRatioDropper(TransformerMixin, BaseEstimator):
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # print(X[self.attribute_names].columns)

        cols_cat = cat_cols(df)
        cats = df[cols_cat]
        # nums = df.drop(columns=cols_cat)
        # cols_num = df[~df[cols_cat]].columns
        cols_num = list(set(df.columns) - set(cols_cat))
        nums = df[cols_num]

        ratio = self.ratio * 100
        missings = missing_ratio_col(nums)
        # print(missings)
        inds = missings[missings['Missing Ratio %'] > ratio].index
        df = df.drop(columns=inds)
        return df


class ColsDropper(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        return df.drop(columns=self.cols)


class DataFrameSelector(TransformerMixin, BaseEstimator):
    def __init__(self, col_names):
        self.attribute_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X[self.attribute_names].columns)

        return X[self.attribute_names].values


class DummyEncoder(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)

        cats = df[cols_cat]
        noncats = df.drop(columns=cols_cat)

        cats = cats.astype('category')
        cats_enc = pd.get_dummies(cats, prefix=cols_cat, dummy_na=True)

        return noncats.join(cats_enc)


# Label encoding is OK when we're using tree models
class MyLabelEncoder(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)

        for col in cols_cat:
            df[col] = df[col].astype('category').cat.add_categories(
                'missing').fillna('missing')
            le = preprocessing.LabelEncoder()
            # TODO add test set together to encoding
            # le.fit(df[col].astype(str).values)
            df[col] = le.fit_transform(df[col].astype(str).values)
        return df


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col + '_FE'
            df[nm] = df[col].map(vc)
            df[nm] = df[nm].astype('float32')
        return df


class CombineEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols_pairs: List[List[str]]):
        self.cols_pairs = cols_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for pair in self.cols_pairs:
            col1 = pair[0]
            col2 = pair[1]
            nm = col1 + '_' + col2
            df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
            df[nm] = df[nm].astype('category')
            # print(nm, ', ', end='')
        return df


class AggregateEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, main_cols: List[str], uids: List[str], aggr_types: List[str],
                 fill_na: bool = True, use_na: bool = False):
        self.main_cols = main_cols
        self.uids = uids
        self.aggr_types = aggr_types
        self.use_na = use_na
        self.fill_na = fill_na

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.main_cols:
            for uid in self.uids:
                for aggr_type in self.aggr_types:
                    col_new = f'{col}_{uid}_{aggr_type}'
                    tmp = df.groupby([uid])[col].agg([aggr_type]).reset_index().rename(
                        columns={aggr_type: col_new})
                    tmp.index = list(tmp[uid])
                    tmp = tmp[col_new].to_dict()
                    df[col_new] = df[uid].map(tmp).astype('float32')
                    if self.fill_na:
                        df[col_new].fillna(-1, inplace=True)
        return df
```

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    # Based on feature engineering from
    # https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600#Encoding-Functions
    ('combine_enc', CombineEncoder(
        [['card1', 'addr1'], ['card1_addr1', 'P_emaildomain']])),
    ('freq_enc', FrequencyEncoder(
        ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])),
    ('aggr_enc', AggregateEncoder(['TransactionAmt', 'D9', 'D11'], [
        'card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'])),

    # Drop columns that have certain high ratio of missing values, and then fill
    # in values e.g. median value. May not be used if using a tree model.
    ('reduce_missing', NumColsRatioDropper(0.5)),
    ('fillna_median', NumColsNaMedianFiller()),

    # Drop some columns that will not be used
    ('drop_cols_basic', ColsDropper(['TransactionID', 'TransactionDT', 'D6',
                                     'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'C3',
                                     'M5', 'id_08', 'id_33', 'card4', 'id_07',
                                     'id_14', 'id_21', 'id_30', 'id_32', 'id_34'])),

    # Drop some columns based on feature importance got from a model.
    ('drop_cols_feat_importance', ColsDropper(
        ['v107', 'v117', 'v119', 'v120', 'v27', 'v28', 'v305'])),

    ('fillna_negative', NumColsNegFiller()),

    # Encode categorical variables. Depending on the kind of model we use,
    # we can choose between label encoding and onehot encoding.
    # ('onehot_enc', DummyEncoder()),
    ('label_enc', MyLabelEncoder()),
])
```

### Split dataset

And as we want to predict future payment fraud based on the past data, so we should not shuffle the dataset when split training and testing sets, but just time-based split.

As this is a imbalanced dataset with 1 class of the target variable have only about 3.5%, so we may want to try sampling methods like over-sampling or SMOTE sampling on the training dataset.

```python
RAND_STATE = 20200119

def data_split_v1(X: pd.DataFrame, y: pd.Series):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, shuffle=False, random_state=RAND_STATE)

    return X_train, X_val, y_train, y_val


def data_split_oversample_v1(X: pd.DataFrame, y: pd.Series):
    from imblearn.over_sampling import RandomOverSampler

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, shuffle=False, random_state=RAND_STATE)

    ros = RandomOverSampler(random_state=RAND_STATE)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    return X_train, X_val, y_train, y_val


def data_split_smoteenn_v1(X: pd.DataFrame, y: pd.Series):
    from imblearn.combine import SMOTEENN

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=RAND_STATE)

    ros = SMOTEENN(random_state=RAND_STATE)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    return X_train, X_val, y_train, y_val
```

# Experiments

Now let's start play with experimenting with simple models like Logistic Regression, or complex models like Gradient Boosting.

Here below is a scaffold for performing experiments.

```python
import os
from datetime import datetime
import json
import pprint

from sklearn import metrics
from sklearn.pipeline import Pipeline
from typing import List, Callable

EXP_DIR = 'exp'

class Experiment:
    def __init__(self, df_nrows: int = None, transform_pipe: Pipeline = None,
                 data_split: Callable = None, model=None, model_class=None,
                 model_param: dict = None):
        self.df_nrows = df_nrows
        self.pipe = transform_pipe

        if data_split is None:
            self.data_split = data_split_v1
        else:
            self.data_split = data_split

        if model_class:
            self.model = model_class(**model_param)
        else:
            self.model = model

        self.model_param = model_param

    def transform(self, X):
        return self.pipe.fit_transform(X)

    def run(self, df_train: pd.DataFrame, save_exp: bool = True) -> float:
        # self.df = load_df(nrows=self.df_nrows)

        y = df_train['isFraud']
        X = df_train.drop(columns=['isFraud'])

        X = self.transform(X)

        X_train, X_val, Y_train, Y_val = self.data_split(X, y)

        # del X
        # gc.collect()

        self.model.fit(X_train, Y_train)

        Y_pred = self.model.predict(X_val)
        self.last_roc_auc = metrics.roc_auc_score(Y_val, Y_pred)

        if save_exp:
            self.save_result()

        return self.last_roc_auc

    def save_result(self, feature_importance:bool=False):
        save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result = {}
        result['roc_auc'] = self.last_roc_auc
        result['transform'] = list(self.pipe.named_steps.keys())
        result['model'] = self.model.__class__.__name__
        result['model_param'] = self.model_param
        result['data_split'] = self.data_split.__name__
        result['num_sample_rows'] = self.df_nrows
        result['save_time'] = save_time
        if feature_importance:
            if hasattr(self.model, 'feature_importances_'):
                result['feature_importances_'] = dict(
                    zip(self.X.columns, self.model.feature_importances_.tolist()))
            if hasattr(self.model, 'feature_importance'):
                result['feature_importances_'] = dict(
                    zip(self.df.columns, self.model.feature_importance.tolist()))

        pprint.pprint(result, indent=4)

        if not os.path.exists(EXP_DIR):
            os.makedirs(EXP_DIR)
        with open(f'{EXP_DIR}/exp_{save_time}_{self.last_roc_auc:.4f}.json', 'w') as f:
            json.dump(result, f, indent=4)

```

```python
import gc


del tran_train, id_train
gc.collect()

df_train = load_df()
```

```python
df_train = load_df()
```

    Mem. usage decreased to 650.48 Mb (66.8% reduction)

## Logistic Regression as baseline

```python
def exp1():
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline(steps=[
        ('combine_enc', CombineEncoder(
            [['card1', 'addr1'], ['card1_addr1', 'P_emaildomain']])),
        ('freq_enc', FrequencyEncoder(
            ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])),
        ('aggr_enc', AggregateEncoder(['TransactionAmt', 'D9', 'D11'], [
         'card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'])),

        ('reduce_missing', NumColsRatioDropper(0.3)),
        ('fillna_median', NumColsNaMedianFiller()),

        ('drop_cols_basic', ColsDropper(['TransactionID', 'TransactionDT', 'C3', 'M5', 'id_08', 'id_33', 'card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34'])),

        # Though onehot encoding is more appropriate for logistic regression, we
        # don't have enough memory to encode that many variables. So we take a
        # step back using label encoding.
        # ('onehot_enc', DummyEncoder()),
        ('label_enc', MyLabelEncoder()),
    ])

    exp = Experiment(transform_pipe=pipe,
                      data_split=data_split_v1,
                      model_class=LogisticRegression,
                      # just use the default hyper paramenters
                      model_param={},
                     )
    exp.run(df_train=df_train)

exp1()
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning:

    lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression



    {   'data_split': 'data_split_v1',
        'model': 'LogisticRegression',
        'model_param': {},
        'num_sample_rows': None,
        'roc_auc': 0.4956463187232307,
        'save_time': '2020-03-26_20-27-08',
        'transform': [   'combine_enc',
                         'freq_enc',
                         'aggr_enc',
                         'reduce_missing',
                         'fillna_median',
                         'drop_cols_basic',
                         'label_enc']}

## Gradient Boosting with LightGBM

Now let's try a Gradient Boosting tree model using the LightGBM implementation, and tune a little on the hyper-parameters to make it a more complex model.

```python
import lightgbm as lgb


class LgbmWrapper:
    def __init__(self, **param):
        self.param = param
        self.trained = None

    def fit(self, X_train, y_train):
        train = lgb.Dataset(X_train, label=y_train)
        self.trained = lgb.train(self.param, train)
        self.feature_importances_ = self.trained.feature_importance()
        return self.trained

    def predict(self, X_val):
        return self.trained.predict(X_val, num_iteration=self.trained.best_iteration)


def exp2():
    pipe = Pipeline(steps=[
        # Based on feature engineering from
        # https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600#Encoding-Functions
        ('combine_enc', CombineEncoder(
            [['card1', 'addr1'], ['card1_addr1', 'P_emaildomain']])),
        ('freq_enc', FrequencyEncoder(
            ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])),
        ('aggr_enc', AggregateEncoder(['TransactionAmt', 'D9', 'D11'], [
            'card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'])),

        # Drop some columns that will not be used
        ('drop_cols_basic', ColsDropper(['TransactionID', 'TransactionDT', 'D6',
                                        'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'C3',
                                        'M5', 'id_08', 'id_33', 'card4', 'id_07',
                                        'id_14', 'id_21', 'id_30', 'id_32', 'id_34'])),

        # Drop some columns based on feature importance got from a model.
        # ('drop_cols_feat_importance', ColsDropper(
        #     ['v107', 'v117', 'v119', 'v120', 'v27', 'v28', 'v305'])),

        ('fillna_negative', NumColsNegFiller()),

        # Label encoding used for tree models.
        # ('onehot_enc', DummyEncoder()),
        ('label_enc', MyLabelEncoder()),
    ])

    param_lgbm = {'objective': 'binary',
                  'boosting_type': 'gbdt',
                  'metric': 'auc',
                  'learning_rate': 0.01,
                  'num_leaves': 2**8,
                  'max_depth': -1,
                  'tree_learner': 'serial',
                  'colsample_bytree': 0.7,
                  'subsample_freq': 1,
                  'subsample': 0.7,
                  'n_estimators': 10000,
                  #  'n_estimators': 80000,
                  'max_bin': 255,
                  'n_jobs': -1,
                  'verbose': -1,
                  'seed': RAND_STATE,
                  # 'early_stopping_rounds': 100,
                  }


    exp = Experiment(transform_pipe=pipe,
                    data_split=data_split_v1,
                     model_class=LgbmWrapper,
                     model_param=param_lgbm,
                     )
    exp.run(df_train=df_train)


exp2()
```

    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:118: UserWarning:

    Found `n_estimators` in params. Will use it instead of argument



    {   'data_split': 'data_split_v1',
        'model': 'LgbmWrapper',
        'model_param': {   'boosting_type': 'gbdt',
                           'colsample_bytree': 0.7,
                           'learning_rate': 0.01,
                           'max_bin': 255,
                           'max_depth': -1,
                           'metric': 'auc',
                           'n_estimators': 10000,
                           'n_jobs': -1,
                           'num_leaves': 256,
                           'objective': 'binary',
                           'seed': 20200119,
                           'subsample': 0.7,
                           'subsample_freq': 1,
                           'tree_learner': 'serial',
                           'verbose': -1},
        'num_sample_rows': None,
        'roc_auc': 0.919589853747652,
        'save_time': '2020-03-27_09-55-43',
        'transform': [   'combine_enc',
                         'freq_enc',
                         'aggr_enc',
                         'drop_cols_basic',
                         'fillna_negative',
                         'label_enc']}

So we got local validation ROC AUC of about 0.9196, this is a looks OK score.

This model's prediction on the test dataset got 0.9398 on publica leader board, and 0.9058 on private leader board. These scores have a somehow big gap to the top scores, but still good enough as there're potentially many ways for improvement. For example, more different ways of transformations and engineering could be performed on the features, try model implementation like CatBoost and XGB, and search for better hyper-parameters. But it assumes you have plenty of computation resource and time.
