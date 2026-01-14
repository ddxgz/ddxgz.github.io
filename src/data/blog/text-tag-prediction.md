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
---

<!-- Go back to [pcx.linkedinfo.co](https://pcx.linkedinfo.co) -->

This code snippet is to predict topic tags based on the text of an article. Each article could have 1 or more tags (usually have at least 1 tag), and the tags are not mutually exclusive. So this is a multi-label classification problem. It's different from multi-class classification, the classes in multi-class classification are mutually exclusive, i.e., each item belongs to 1 and only 1 class.

In this snippet, we will use `OneVsRestClassifier` (the One-Vs-the-Rest) in scikit-learn to process the multi-label classification. The article data will be retrieved from [LinkedInfo.co](https://linkedinfo.co) via Web API.

1. Preprocessing data and explore the method
2. Search for best model include SVM and Neural Networks (Will be updated)
3. Training and testing with the best parameters

The methods in this snippet should give credits to [Working With Text Data - scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Preprocessing data and explore the method

`dataset.df_tags` fetches the data set from [LinkedInfo.co](https://linkedinfo.co). It calls Web API of LinkedInfo.co to retrieve the article list, and then download and extract the full text of each article based on an article's url. The tags of each article are encoded using `MultiLabelBinarizer` in scikit-learn. The implementation of the code could be found in [dataset.py](https://github.com/ddxgz/linkedinfo-ml-models/blob/master/dataset.py). We've set the parameter of `content_length_threshold` to 100 to screen out the articles with less than 100 for the description or full text.

```python
import dataset

ds = dataset.df_tags(content_length_threshold=100)
```

The dataset contains 3353 articles by the time retrieved the data.
The dataset re returned as an object with the following attribute:

> - ds.data: pandas.DataFrame with cols of title, description, fulltext
> - ds.target: encoding of tagsID
> - ds.target_names: tagsID
> - ds.target_decoded: the list of lists contains tagsID for each info

```
ds.data.head()
ds.target[:5]
ds.target_names[:5]
ds.target_decoded[:5]
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

Now we've transformed the target (tags) but we cannot directly perform the algorithms on the text data, so we have to process and transform them into vectors. In order to do this, we will use `TfidfVectorizer` to preprocess, tokenize, filter stop words and transform the text data. The `TfidfVectorizer` implements the [_tf-idf_](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (Term Frequency-Inverse Deocument Frequency) to reflect how important a word is to to a docuemnt in a collection of documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Use the default parameters for now, use_idf=True in default
vectorizer = TfidfVectorizer()
# Use the short descriptions for now for faster processing
X = vectorizer.fit_transform(df_data.description)
X.shape
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

```
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
    print(f'info title: {title} \ntags: {t} \npredicted as: {p}')

```

```
info title: Designing resilient systems: Circuit Breakers or Retries? (Part 1)
tags: ('circuit-breaker', 'system-architecture')
predicted as: ('golang', 'system-architecture')
info title: Zuul 2 : The Netflix Journey to Asynchronous, Non-Blocking Systems
tags: ('asynchronous', 'system-architecture')
predicted as: ()
info title: 对称加密算法与非对称加密算法的优缺点
tags: ('crytography',)
predicted as: ()
info title: Go coding in go way
tags: ('golang',)
predicted as: ('golang',)
info title: Emacs Doom for Newbies
tags: ('doom-emacs', 'emacs')
predicted as: ('emacs',)
info title: Basic testing patterns in Go
tags: ('golang', 'testing')
predicted as: ('golang', 'testing')
info title: Dead Simple Python: Virtual Environments and pip
tags: ('python', 'virtualenv')
predicted as: ('python',)
info title:  Overcoming RESTlessness
tags: ('api', 'restful', 'web')
predicted as: ('api',)
info title: Let’s Build A Simple Interpreter. Part 7.
tags: ('compiler', 'interpreter', 'pascal', 'python')
predicted as: ('compiler', 'interpreter', 'pascal', 'python')
info title: Spring RestTemplate详解
tags: ('html', 'java', 'spring')
predicted as: ('java',)
```

Here above is a fraction of the list that shows the manually input tags and the predicted tags. We can see that usually the more frequently appeared and more popular tags have better change to be correctly predicted. Personally, I would say the prediction is satisfied to me comparing when I tag the articles manually. However, there's much room for improvement.

<!-- However, the manual tags, as the training and prediction comparing group, suffer from problems like  -->
