## LIBRARIES
import matplotlib as mpl
import numpy as np
from pandas import DataFrame, Series, read_csv, concat, merge
from sklearn import preprocessing
print("libraries loaded")

# CALCULATES REMAINING USUFULL LIFE RUL COLUMN
# Get max cycle per equipment
def calculate_rul(m_df):
    d1 = m_df.groupby('id').max()
    # Select id and cycle columns only
    cycle_df = [d1['cycle'][i+1] for i in range(len(d1))]
    id_df = [d1.index[i] for i in range(len(d1))]
    # Add max cycle to train_df
    d2 = concat([DataFrame(id_df), DataFrame(cycle_df)], axis=1)
    d2.columns = ['id', 'max']
    # Calculate remaining usefull life up to 0 when equipment breaks. This is the Y
    d3 = merge(m_df, d2, on='id' )
    d3['RUL'] = d3['max']-d3['cycle']
    d3 = d3.drop(['max'], axis=1)
    print('RUL calculated done')
    return(d3[['id','RUL']])

# CALCULATE BREAK OR NO-BREAK COLUMN FOR BINARY CLASSIFICATON
def calculate_binary(m_df):
    result = lambda row: 1 if row['RUL']<=30 else 0  # calculates row
    m_df['Binary'] = m_df.apply(result, axis=1) # passes axis1 to function
    print('Binary calculated done')
    return(m_df[['Binary']])

# CALCULATE BREAK OR NO BREAK FOR MULTI CLASS CLASSIFICATION
def calculate_multiclass(m_df):
    result = lambda row: 2 if row['RUL']<=15 else (1 if row['RUL']<=30 else 0)
    m_df['Multi'] = m_df.apply(result, axis=1)
    print('multi-class calculated done')
    return(m_df[['Multi']])

# FEATURE ENGINEERING. CALCULATES STARDARD DEVIATION AND MEAN FOR EACH SENSOR READING
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

# TRANSFORMATION. CREATE A TRANSFORMATION PACKAGEDATA 
def transform_scale(m_df):
    equipment_df = m_df.drop(['id'], axis=1) # drop id
    min_max_scaler = preprocessing.MinMaxScaler()
    fit_scale = min_max_scaler.fit(equipment_df)
    return(fit_scale)

#%% NORMILZE FEATURES
# normalize dataset with a transformation
def normalize_features(m_df, transformation):
    equipment_df = m_df.drop(['id'], axis=1) # drop id and rul columns
    equipment_df = DataFrame(transformation.transform(equipment_df), 
                    columns=equipment_df.columns, index=equipment_df.index)
    print('Data Set engineered completed')
    return(equipment_df)

#%% SELECT MAX CYCLE
def select_max(m_df):
    idx = m_df.groupby('id')['cycle'].transform(max) == m_df['cycle'] # index max cycle by equipment id
    m_df = m_df[idx] # filter set with last cycle
    return(m_df)