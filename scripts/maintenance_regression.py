#%%
## LIBRARIES
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pandas import DataFrame, Series, read_csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import display, HTML
print("libraries loaded")

#%% IMPORT DATA
train_df = pd.read_csv('./data/train_readings.csv')
test_df = pd.read_csv('./data/test_readings.csv')
test_y = pd.read_csv('./data/test_labels.csv')

#%% CREATE REMAINING USUFULL LIFE RUL
# Get max cycle per equipment
def calculate_rul(m_df):
    d1 = m_df.groupby('id').max()
    # Select id and cycle columns only
    cycle_df = [d1['cycle'][i+1] for i in range(len(d1))]
    id_df = [d1.index[i] for i in range(len(d1))]
    # Add max cycle to train_df
    d2 = pd.concat([DataFrame(id_df), DataFrame(cycle_df)], axis=1)
    d2.columns = ['id', 'max']
    # Calculate remaining usefull life up to 0 when equipment breaks. This is the Y
    d3 = pd.merge(m_df, d2, on='id' )
    d3['RUL'] = d3['max']-d3['cycle']
    d3 = d3.drop(['max'], axis=1)
    print('RUL calculated done')
    return(d3)

#%% FEATURE ENGINEERING
def engineer_features(m_df, window):
    final_df = DataFrame() # stores the complete results
    result_df = DataFrame() # stores sd and mu values per equipment
    window_size = window # 5 cycles rolling window

    # for each equipment calculte their sensors sd and mu
    for a in range(100):
        condition = m_df['id']==a+1 # filter equipment
        d_temp = m_df[condition].loc[:,'s1':'s21'] # select only sensor data
        result_df = DataFrame() # reset df

        # calculate sd and mean for each sensor
        for i in range(len(d_temp.columns)):
            result_df['sd'+str(i+1)] = (d_temp['s'+str(i+1)].rolling(window_size, min_periods=1).std())
            result_df['mu'+str(i+1)] = (d_temp['s'+str(i+1)].rolling(window_size, min_periods=1).mean())
        final_df = final_df.append(result_df) # append equipment results

    final_df = final_df.fillna(0) # first sd is NA
    print('Feature engineering done')
    return(final_df)

#%% TRANSFORMATION
# Create a transformation 
def transform_scale(m_df):
    equipment_df = m_df.drop(['id', 'RUL'], axis=1) # drop id
    min_max_scaler = preprocessing.MinMaxScaler()
    fit_scale = min_max_scaler.fit(equipment_df)
    return(fit_scale)

#%% NORMILZE FEATURES
# normalize dataset with a transformation
def normalize_features(m_df, transformation):
    equipment_df = m_df.drop(['id', 'RUL'], axis=1) # drop id and rul columns
    equipment_df = DataFrame(transformation.transform(equipment_df), 
                    columns=equipment_df.columns, index=equipment_df.index)
    print('Data Set engineered completed')
    return(equipment_df)


#%% SELECT MAX CYCLE
def select_max(m_df):
    idx = m_df.groupby('id')['cycle'].transform(max) == m_df['cycle'] # index max cycle by equipment id
    m_df = m_df[idx] # filter set with last cycle
    return(m_df)
    
#%% PREPARE TRAIN DATA
# run steps to prepare the Training dataset
def prepare_train(m_df):
    rul_df = calculate_rul(m_df) # Step 1. Calculate RUL test_y
    eng_df = engineer_features(rul_df, 15) # Step 2. Engineer features. Adds mean and sd
    whole_df = pd.concat([rul_df, eng_df], axis=1) # Step 3. Adds new features to dataset
    y_train = whole_df['RUL'] # Step 4. Gets y
    scale = transform_scale(whole_df) # Step 5. Fit transformation with train dataset. 
    norm_df = normalize_features(whole_df, scale) # Step 6. Normalize train data set
    print('Data preparation done')
    return(norm_df, y_train, scale)

    
#%% PREPARE TEST DATA
# Uses the scaling transformation from the training
def prepare_test(m_df, scale):
    rul_df = calculate_rul(m_df) # Step 1. Calculates Rul
    eng_df = engineer_features(rul_df, 15) # Step 2. Engineer Features. mean and sd
    whole_df = pd.concat([rul_df, eng_df], axis=1) # Step 3. Add new features to dataset
    whole_df = select_max(whole_df) # Step 4. Get's only the latest cycle 
    norm_df = normalize_features(whole_df, scale) # Step 5. Normalizes dataset
    print('Data preparation done')
    return(norm_df)


# ************** TRAIN AND TEST ************
#%% PREPARE TRAIN AND TEST DATA SETS
train_prep_df, y_train, scale = prepare_train(train_df)
test_prep_df, y_test = prepare_test(test_df, scale), test_y

#%%
print(train_df.colums)

#%% FIT REGRESSION MODELS WITH SOME BASIC PARAMETERS
models = [('Random Forest', RandomForestRegressor(n_estimators=35, min_samples_split=2)),
        ('Decision Tree', DecisionTreeRegressor(max_depth=3, random_state=0)),
        ('Bagging Tree', BaggingRegressor()),
        ('Linear Regression', LinearRegression())]
train_results, models_name, test_results = [], [], []

for name, model in models:
    # Step 1. Kfold to get a first idea on performance
    kfold = model_selection.KFold(n_splits=10, random_state=8)
    cv_results = model_selection.cross_val_score(model, train_prep_df, y_train, cv=kfold)
    
    # Step 2. Compare to a full run training and testing
    model.fit(train_prep_df, y_train)
    train_pred = model.predict(train_prep_df)
    test_pred = model.predict(test_prep_df)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test.values, test_pred))
    train_results.append(train_rmse)
    test_results.append(test_rmse)
    models_name.append(name)
    print("%s, cv result:%f, train:%f, test:%f" %(name, cv_results.mean(), train_rmse, test_rmse))

# RESULTS. All of them pretty similar on test.
#Random Forest, cv result:0.683595, train:5.227035, test:27.921773
#Decision Tree, cv result:0.659665, train:38.965967, test:30.493803
#Bagging Tree, cv result:0.660395, train:6.891489, test:30.558837
#Linear Regression, cv result:0.664048, train:39.243980, test:29.707504

#%% SELECT RANDOM FOREST AND TUNE NUMBER OF TREES
train_rmses, test_rmses = [], []
for i in range(50):
    random_forest = RandomForestRegressor(n_estimators=i+1, random_state=2)
    random_forest.fit(train_prep_df, y_train)
    train_pred = random_forest.predict(train_prep_df)
    test_pred = random_forest.predict(test_prep_df)
    train_rmses.append(np.sqrt(mean_squared_error(y_train.values, train_pred)))
    test_rmses.append(np.sqrt(mean_squared_error(y_test.values, test_pred)))

plt.plot(train_rmses, color='red')
plt.plot(test_rmses, color='blue')
plt.show()

#%% 
print(test_results)
print(train_results)
print(models_name)

#%% PLOT RESULTS
cycles = DataFrame(select_max(test_df[['id','cycle']])['cycle'].values)
results = pd.concat([cycles, y_test, DataFrame(tree_test_pred)], axis=1)
results.columns = ['cycle', 'RUL true', 'RUL pred']
results = results.sort_values(by=['cycle'], ascending=[1])
results.to_csv('./data/result_predictions.csv')

plt.plot(results['RUL true'].values, color='green')
plt.plot(results['RUL pred'].values, color='blue')
plt.plot(results['cycle'].values, color='red')

plt.show()

