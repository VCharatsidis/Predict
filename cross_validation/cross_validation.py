from sklearn.model_selection import RandomizedSearchCV

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from get_input import get_input

train_one_hot, train_features, train_labels = get_input()

# Number of trees in random forest
n_estimators = [2000]  #[int(x) for x in np.linspace(start=200, stop=2000, num=20)]

# Number of features to consider at every split
max_features = [int(x) for x in np.linspace(1, len(train_features[0]), num=len(train_features[0]))]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 32, num=27)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(5, 32, num=27)]

# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(5, 32, num=27)]

# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }

print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
numeric_rf = RandomForestClassifier()

# Random search of parameters, using cv fold cross validation,
# search across n_iter different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=numeric_rf, param_distributions=random_grid, n_iter=1000, cv=8, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model

print(len(train_features))

counter = 0
for i in train_features:
    print(i)
    print(str(counter)+" "+str(train_labels[counter]))
    counter += 1

rf_random.fit(train_features, train_labels)

print(rf_random.best_params_)


