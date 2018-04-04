
#%% LIBRARIES
#import pandas as pd
from scripts import libraries
from pandas import read_csv, concat, DataFrame
from numpy import sqrt
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
print('Libraries done')

#%% DATA
train_df = read_csv('./data/train_readings.csv')
test_df = read_csv('./data/test_readings.csv')
test_y = read_csv('./data/test_labels.csv')

#%% PREPARE TRAIN DATASET
def prep_train(m_df):
    rul = libraries.calculate_rul(m_df)
    multi = libraries.calculate_multiclass(rul)
    engineered_df = libraries.engineer_features(m_df,15) # Step 2. Calculate SD and MU for sensors only
    whole_df = concat([m_df, engineered_df], axis=1) # Step 3. Adds new features to dataset
    scale = libraries.transform_scale(whole_df) # Step 5. Fit transformation with train dataset.
    norm_df = libraries.normalize_features(whole_df, scale) # Step 6. Normalizes data set
    print('Train Data preparation done')
    return(norm_df, multi, scale)

#%% PREPARE TEST DATASET
# Uses the scaling transformation from the training
def prepare_test(m_df, scale):
    eng_df = libraries.engineer_features(m_df, 15) # Step 2. Engineer Features. mean and sd
    whole_df = concat([m_df, eng_df], axis=1) # Step 3. Add new features to dataset
    whole_df = libraries.select_max(whole_df) # Step 4. Get's only the latest cycle 
    norm_df = libraries.normalize_features(whole_df, scale) # Step 5. Normalizes dataset
    print('Test Data preparation done')
    return(norm_df)


########## TRAIN AND TEST MODELS ###########
#%% 
train_prep_df, y_train, scale = prep_train(train_df)
test_prep_df, y_test = prepare_test(test_df, scale), libraries.calculate_multiclass(test_y)

#%% REGRESSION MODELS WITH SOME BASIC PARAMETERS
models = [('Random Forest', RandomForestClassifier(n_estimators=8, min_samples_split=12)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=32, random_state=128)),
        ('AdaBoost', AdaBoostClassifier()),
        ('LinearSVC', LinearSVC(random_state=0))]
train_results, models_name, test_results = [], [], []

#%% MODELS
for name, model in models:
    # Step 1. Kfold to get a first idea on performance
    #kfold = model_selection.KFold(n_splits=10, random_state=8)
    #cv_results = model_selection.cross_val_score(model, train_prep_df, y_train, cv=kfold)
    
    # Step 2. Compare to a full run training and testing
    model.fit(train_prep_df, y_train['Multi'])
    train_pred = model.predict(train_prep_df)
    test_pred = model.predict(test_prep_df)
    train_acurracy = accuracy_score(y_train['Multi'], train_pred)
    test_acurracy = accuracy_score(y_test['Multi'], test_pred)
    train_results.append(train_acurracy)
    test_results.append(test_acurracy)
    models_name.append(name)
    print("%s, train:%f, test:%f" %(name, train_acurracy, test_acurracy))