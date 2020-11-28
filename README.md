# BoxClassifier

### Description
An original classification algorithm I wrote based off of partitioning the feature space into equal sizes and finding the dominant class in each partition.

### Limitations
* Currently only supports binary classification problems. Will expand in the near future.
* Currently only accepts numpy arrays as inputs; this is easy to fix but I'm lazy at the moment.

### Dependencies
* numpy, pandas

### Hyperparameters
* partitions (default = 20)
  * This parameter controls how many times to partition the feature space. Depending on the dataset, this parameter can cause a lot of variance. It's recommended to explore a wide range of values, e.g from 5 --> 200.
  
### Creating the object
```python
bc = BoxClassifier(100)
bc.fit(xtrain, ytrain) # numpy arrays
bc.predict(xtest)
```

### Example
I used sklearn's make_classification to test performance on synthetic data sets. \
I'm using ```random_state = 42``` as a seed.

```python
sample_data = make_classification(n_features = 100, n_samples = 1000, n_informative = 15, n_redundant = 40, class_sep = 0.25, random_state = 42)
xtrain, xtest, ytrain, ytest = train_test_split(sample_data[0], sample_data[1], test_size = 0.3, random_state = 42)
```

BoxClassifier has support for sklearn's cross_val_score for cross validation metrics.

```python
bc_cv = cross_val_score(BoxClassifier(partitions = 20), xtrain, ytrain, scoring='f1', cv=5).mean()
lr_cv = cross_val_score(LogisticRegression(max_iter=2500), xtrain, ytrain, scoring='f1', cv=5).mean()
nb_cv = cross_val_score(GaussianNB(), xtrain, ytrain, scoring='f1', cv=5).mean()
rf_cv = cross_val_score(RandomForestClassifier(), xtrain, ytrain, scoring='f1', cv=5).mean()
svc_cv = cross_val_score(SVC(), xtrain, ytrain, scoring='f1', cv=5).mean()
```

Below is a comparison between popular algorithms with 5-CV on the above synthetic data set.

```python
print("Box Classifier: \t", round(bc_cv,3))         # 0.828
print("Logistic Regression: \t", round(lr_cv,3))    # 0.571
print("Gaussian Naive Bayes: \t", round(nb_cv,3))   # 0.712
print("Random Forest: \t\t", round(rf_cv,3))        # 0.788
print("SVC: \t\t\t", round(svc_cv,3))               # 0.867
```

We can see that (in this case) box classifier is only surpassed by SVC with both having a very decent F1 score.
