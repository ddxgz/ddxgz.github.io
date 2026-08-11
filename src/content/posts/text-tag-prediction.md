---
title: "Multi-label classification to predict topic tags of technical articles from LinkedInfo.co"
description: "Multi-label topic tag prediction for LinkedInfo.co using TF-IDF features and LinearSVC."
pubDatetime: 2019-09-11T11:36:43Z
modDatetime: 2019-09-13T11:36:43Z
tags:
  [
    "Machine Learning",
    "Multi-label classification",
    "Text classification",
    "LinkedInfo.co",
  ]
canonicalURL: "https://pcx.linkedinfo.co/post/text-tag-prediction/"
links:
  - name: "kaggle Notebook"
    url: "https://www.kaggle.com/pcx813/multi-label-classification-for-article-tags-svm"
---

<!-- Go back to [pcx.linkedinfo.co](https://pcx.linkedinfo.co) -->

This code snippet is to predict topic tags based on the text of an article. Each article could have 1 or more tags (usually have at least 1 tag), and the tags are not mutually exclusive. So this is a multi-label classification problem. It's different from multi-class classification, the classes in multi-class classification are mutually exclusive, i.e., each item belongs to 1 and only 1 class.

In this snippet, we will use `OneVsRestClassifier` (the One-Vs-the-Rest) in scikit-learn to process the multi-label classification. The article data will be retrieved from [LinkedInfo.co](https://linkedinfo.co) via Web API. The methods in this snippet should give credits to [Working With Text Data - scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) and [this post](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff).

## Table of contents

## Preprocessing data and explore the method

`dataset.df_tags` fetches the data set from [LinkedInfo.co](https://linkedinfo.co). It calls Web API of LinkedInfo.co to retrieve the article list, and then download and extract the full text of each article based on an article's url. The tags of each article are encoded using `MultiLabelBinarizer` in scikit-learn. The implementation of the code could be found in [dataset.py](https://github.com/ddxgz/linkedinfo-ml-models/blob/master/dataset.py). We've set the parameter of `content_length_threshold` to 100 to screen out the articles with less than 100 for the description or full text.

```python
import dataset

ds = dataset.df_tags(content_length_threshold=100)
```

The dataset contains 3353 articles by the time retrieved the data.
The dataset re returned as an object with the following attribute:

- ds.data: pandas.DataFrame with cols of title, description, fulltext
- ds.target: encoding of tagsID
- ds.target_names: tagsID
- ds.target_decoded: the list of lists contains tagsID for each info

```python
>> ds.data.head()
```

|     | description                                                                     | fulltext                                                              | title                                             |
| --- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| 0   | Both HTTP 1.x and HTTP/2 rely on lower level c...                               | \[Stressgrid\](/)\n\n\_\_\n\n[]\(/ "home")\n\n \* \[...               | Achieving 100k connections per second with Elixir |
| 1   | At Phusion we run a simple multithreaded HTTP ...                               | \[\!\[Hongli Lai](/images/avatar-b64f1ad5.png)](/...                  | What causes Ruby memory bloat?                    |
| 2   | Have you ever wanted to contribute to a projec...                               | [ ![Real Python](/static/real-python-logo.ab1a...                     | Managing Multiple Python Versions With pyenv      |
| 3   | 安卓在版本Pie中第一次引入了ART优化配置文件，这个新特性利用发送到Play Cloud的... | 安卓在版本Pie中第一次引入了\[ART优化配置文件\](https://youtu.be/Yi... | ART云配置文件，提高安卓应用的性能                 |
| 4   | I work at Red Hat on GCC, the GNU Compiler Col...                               | \[ \![Red Hat\nLogo](https://developers.redhat.c...                   | Usability improvements in GCC 9                   |

```python
>> ds.target[:5]
```

```text
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]])
```

```python
>> ds.target_names[:5]
```

```text
array(['academia', 'access-control', 'activemq', 'aes', 'agile'],
      dtype=object)
```

```python
>> ds.target_decoded[:5]
```

```text
[['concurrency', 'elixir'],
 ['ruby'],
 ['python', 'virtualenv'],
 ['android'],
 ['gcc']]
```

The following snippet is the actual process of getting the above dataset, by
reading from file.

```python
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

infos_file = 'data/infos/infos_0_3353_fulltext.json'
with open(infos_file, 'r') as f:
    infos = json.load(f)

content_length_threshold = 100

data_lst = []
tags_lst = []
for info in infos['content']:
    if len(info['fulltext']) < content_length_threshold:
        continue
    if len(info['description']) < content_length_threshold:
        continue
    data_lst.append({'title': info['title'],
                     'description': info['description'],
                     'fulltext': info['fulltext']})
    tags_lst.append([tag['tagID'] for tag in info['tags']])

df_data = pd.DataFrame(data_lst)
df_tags = pd.DataFrame(tags_lst)

# fit and transform the binarizer
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tags_lst)
Y.shape
```

```text
(3221, 560)
```

Now we've transformed the target (tags) but we cannot directly perform the algorithms on the text data, so we have to process and transform them into vectors. In order to do this, we will use `TfidfVectorizer` to preprocess, tokenize, filter stop words and transform the text data. The `TfidfVectorizer` implements the [_tf-idf_](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (Term Frequency-Inverse Document Frequency) to reflect how important a word is to to a document in a collection of documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Use the default parameters for now, use_idf=True in default
vectorizer = TfidfVectorizer()
# Use the short descriptions for now for faster processing
X = vectorizer.fit_transform(df_data.description)
X.shape
```

```text
(3221, 35506)
```

As mentioned in the beginning, this is a multi-label classification problem, we will use `OneVsRestClassifier` to tackle our problem. And firstly we will use the SVM (Support Vector Machines) with linear kernel, implemented as `LinearSVC` in scikit-learn, to do the classification.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Use default parameters, and train and test with small set of samples.
clf = OneVsRestClassifier(LinearSVC())

from sklearn.utils import resample

X_sample, Y_sample = resample(
    X, Y, n_samples=1000, replace=False, random_state=7)

# X_sample_test, Y_sample_test = resample(
#     X, Y, n_samples=10, replace=False, random_state=1)

X_sample_train, X_sample_test, Y_sample_train, Y_sample_test = train_test_split(
    X_sample, Y_sample, test_size=0.01, random_state=42)

clf.fit(X_sample, Y_sample)
Y_sample_pred = clf.predict(X_sample_test)

# Inverse transform the vectors back to tags
pred_transformed = mlb.inverse_transform(Y_sample_pred)
test_transformed = mlb.inverse_transform(Y_sample_test)

for (t, p) in zip(test_transformed, pred_transformed):
    print(f'tags: {t} predicted as: {p}')
```

```text
tags: ('javascript',) predicted as: ('javascript',)
tags: ('erasure-code', 'storage') predicted as: ()
tags: ('mysql', 'network') predicted as: ()
tags: ('token',) predicted as: ()
tags: ('flask', 'python', 'web') predicted as: ()
tags: ('refactoring',) predicted as: ()
tags: ('emacs',) predicted as: ()
tags: ('async', 'javascript', 'promises') predicted as: ('async', 'javascript')
tags: ('neural-networks',) predicted as: ()
tags: ('kubernetes',) predicted as: ('kubernetes',)
```

Though not very satisfied, this classifier predicted right a few tags. Next we'll try to search for the best parameters for the classifier and train with fulltext of articles.

## Search for best model parameters for SVM with linear kernel

For the estimators `TfidfVectorizer` and `LinearSVC`, they both have many parameters could be tuned for better performance. We'll the `GridSearchCV` to search for the best parameters with the help of `Pipeline`.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


# Split the dataset into training and test set, and use fulltext of articles:
X_train, X_test, Y_train, Y_test = train_test_split(
    df_data.fulltext, Y, test_size=0.5, random_state=42)

# Build vectorizer classifier pipeline
clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

# Grid search parameters, I minimized the parameter set based on previous
# experience to accelerate the processing speed.
# And the combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True
parameters = {
    'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
    'vect__max_df': [1, 0.9, 0.8, 0.7],
    'vect__min_df': [1, 0.9, 0.8, 0.7, 0],
    'vect__use_idf': [True, False],
    'clf__estimator__penalty': ['l1', 'l2'],
    'clf__estimator__C': [1, 10, 100, 1000],
    'clf__estimator__dual': [False],
}

gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
gs_clf.fit(X_train, Y_train)
```

```python
import datetime
from sklearn import metrics


# Predict the outcome on the testing set in a variable named y_predicted
Y_predicted = gs_clf.predict(X_test)

print(metrics.classification_report(Y_test, Y_predicted))
print(gs_clf.best_params_)
print(gs_clf.best_score_)

# Export some of the result cols
cols = [
    'mean_test_score',
    'mean_fit_time',
    'param_vect__ngram_range',
]
df_result = pd.DataFrame(gs_clf.cv_results_)
df_result = df_result.sort_values(by='rank_test_score')
df_result = df_result[cols]

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
df_result.to_html(
    f'data/results/gridcv_results_{timestamp}_linearSVC.html')
```

Here we attach the top-5 performed classifiers with selected parameters.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank_test_score</th>
      <th>mean_test_score</th>
      <th>mean_fit_time</th>
      <th>param_vect__max_df</th>
      <th>param_vect__ngram_range</th>
      <th>param_vect__use_idf</th>
      <th>param_clf__estimator__penalty</th>
      <th>param_clf__estimator__C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>1</td>
      <td>0.140811</td>
      <td>96.127405</td>
      <td>0.8</td>
      <td>(1, 4)</td>
      <td>True</td>
      <td>l1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2</td>
      <td>0.140215</td>
      <td>103.252332</td>
      <td>0.7</td>
      <td>(1, 4)</td>
      <td>True</td>
      <td>l1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2</td>
      <td>0.140215</td>
      <td>98.990952</td>
      <td>0.9</td>
      <td>(1, 4)</td>
      <td>True</td>
      <td>l1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>154</th>
      <td>2</td>
      <td>0.140215</td>
      <td>1690.433151</td>
      <td>0.9</td>
      <td>(1, 4)</td>
      <td>True</td>
      <td>l1</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>68</th>
      <td>5</td>
      <td>0.139618</td>
      <td>70.778621</td>
      <td>0.7</td>
      <td>(1, 3)</td>
      <td>True</td>
      <td>l1</td>
      <td>10</td>
    </tr>
  </tbody>
</table>

## Training and testing with the best parameters

Based on the grid search results, we found the following parameters combined with the default parameters have the best performance. Now let's see how it will perform.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    df_data, Y, test_size=0.2, random_state=42)

clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True,
                             max_df=0.8, ngram_range=[1, 4])),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l1', C=10, dual=False))),
])

clf.fit(X_train.fulltext, Y_train)


Y_pred = clf.predict(X_test.fulltext)

# Inverse transform the vectors back to tags
pred_transformed = mlb.inverse_transform(Y_pred)
test_transformed = mlb.inverse_transform(Y_test)

for (title, t, p) in zip(X_test.title, test_transformed, pred_transformed):
    print(f'Article title: {title} \n'
          f'Manual tags:  {t} \n'
          f'predicted as: {p}\n')
```

Here below is a fraction of the list that shows the manually input tags and the predicted tags. We can see that usually the more frequently appeared and more popular tags have better change to be correctly predicted. Personally, I would say the prediction is satisfied to me comparing when I tag the articles manually. However, there's much room for improvement.

<!-- However, the manual tags, as the training and prediction comparing group, suffer from problems like  -->

```text
Article title: Will PWAs Replace Native Mobile Apps?
Manual tags:  ('pwa',)
predicted as: ('pwa',)

Article title: 基于Consul的分布式信号量实现
Manual tags:  ('consul', 'distributed-system')
predicted as: ('microservices', 'multithreading')

Article title: commit 和 branch 理解深入
Manual tags:  ('git',)
predicted as: ('git',)

Article title: Existential types in Scala
Manual tags:  ('scala',)
predicted as: ('scala',)

Article title: Calling back into Python from llvmlite-JITed code
Manual tags:  ('jit', 'python')
predicted as: ('compiler', 'python')

Article title: Writing a Simple Linux Kernel Module
Manual tags:  ('kernel', 'linux')
predicted as: ('linux',)

Article title: Semantic segmentation with OpenCV and deep learning
Manual tags:  ('deep-learning', 'opencv')
predicted as: ('deep-learning', 'image-classification', 'opencv')

Article title: Transducers: Efficient Data Processing Pipelines in JavaScript
Manual tags:  ('javascript',)
predicted as: ('javascript',)

Article title: C++之stl::string写时拷贝导致的问题
Manual tags:  ('cpp',)
predicted as: ('functional-programming',)

Article title: WebSocket 浅析
Manual tags:  ('websocket',)
predicted as: ('websocket',)

Article title: You shouldn’t name your variables after their types for the same reason you wouldn’t name your pets “dog” or “cat”
Manual tags:  ('golang',)
predicted as: ('golang',)

Article title: Introduction to Data Visualization using Python
Manual tags:  ('data-visualization', 'python')
predicted as: ('data-visualization', 'matplotlib', 'python')

Article title: How JavaScript works: A comparison with WebAssembly + why in certain cases it’s better to use it over JavaScript
Manual tags:  ('javascript', 'webassembly')
predicted as: ('javascript', 'webassembly')

Article title: Parsing logs 230x faster with Rust
Manual tags:  ('log', 'rust')
predicted as: ('rust',)

Article title: Troubleshooting Memory Issues in Java Applications
Manual tags:  ('java', 'memory')
predicted as: ('java',)

Article title: How to use Docker for Node.js development
Manual tags:  ('docker', 'node.js')
predicted as: ('docker',)
```

## A glance at the different evaluation metrics

Now let's have a look at the evaluation metrics on the prediction performance. Evaluating multi-label classification is very different from evaluating binary classification. There're quite many different evaluation methods for different situations in [the model evaluation part of scikit-learn's documentation](https://scikit-learn.org/stable/modules/model_evaluation.html). We will take a look at the ones that suit this problem.

We can start with the `accuracy_score` function in `metrics` module. As mentioned in scikit-learn documentation, in multi-label classification, a subset accuracy is 1.0 when the entire set of predicted labels for a sample matches strictly with the true label set. The equation is simple like this:

$$\operatorname{accuracy}(y, \hat{y})=\frac{1}{n\_{\text {samples }}} \sum\_{i=0}^{n\_{\text {samples }}-1} 1\left(\hat{y}\_{i}=y\_{i}\right)$$

```python
from sklearn import metrics
import matplotlib.pyplot as plt

metrics.accuracy_score(Y_test, Y_pred)
```

```text
0.26356589147286824
```

The score is somehow low. But we should be noted that for this problem, an inexact match of the labels is acceptable in many cases, e.g., an article talks about the golang's interface is predicted with an only label `golang` while it was manually labeled with `golang` and `interface`. So to my opinion, this `accuracy_score` is not a good evaluation metric for this problem.

Now let's see the `classification_report` that presents averaged precision, recall and f1-score.

```python
print(metrics.classification_report(Y_test, Y_pred))
```

|          |     | precision | recall | f1-score | support |
| -------- | --- | --------- | ------ | -------- | ------- |
| micro    | avg | 0.74      | 0.42   | 0.54     | 1186    |
| macro    | avg | 0.17      | 0.13   | 0.14     | 1186    |
| weighted | avg | 0.60      | 0.42   | 0.48     | 1186    |

Let's look at the _micro_ row. Why? Let me quote scikit-learn's documentation:

> "micro" gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored.

Here we're more interested in the average precision, which is 0.74. As we mentioned, for this problem and for me, it's more important to not predict a label that should be negative to an article. Some of the labels for an article, e.g., the label `interface` for the just mentioned article, are less important. So I'm OK for having a low score of recall, which measure how good the model predicts all the labels as the manually labeled.

However, there's much room for improvement.

- Many of the labels have very few appearances or even once. These labels could be filtered out or oversampling with text augmentation to mitigate the impact to model performance.
- The training-test set split should be controlled by methods like stratified sampling, so that all the labels would appear in both sets with similar percentages. But again this problem is unlikely to be solved by now since there isn't enough samples.
- Another problem to be though about is, the training samples are not equally labeled, i.e., for the same example all the articles talking about golang's interface, some of them labeled with `golang` + `interface` while some of them labeled only `golang`.
