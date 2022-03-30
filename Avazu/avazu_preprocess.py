from math import floor, log
import pandas as pd
from sklearn.preprocessing import LabelEncoder  

def avazu_prep(data_path):
  av_columns = ['click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain'
  ,'app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
  data_features = ['hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain'
  ,'app_category','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21','user']

  #Read data, not using the id column in this case (should also skip hour in small case)
  data = pd.read_csv(data_path, delimiter=',', usecols =  av_columns
  , engine = 'python')

  #Define user category:
  data['user'] = data.apply(lambda x: str(x['device_id']) + str(x['device_ip']) + str(x['device_model']), axis = 1)
  data.drop(columns=['device_id','device_ip','device_model'], inplace=True)

  #Remove low count features (15)
  string_features = ['site_id','site_domain','site_category','app_id','app_domain','app_category','user']
  to_skip = ['user', 'C18','hour']
  low_count = 15
  for col in data_features:
    if col in to_skip: #Consider to maybe skip also other features
      continue
    counts = data.groupby(col).size().sort_values(ascending=False)
    idx = len(counts)
    for val, count in counts.items():
      if count < int(low_count): 
        counts = counts.tail(idx).index.tolist()
        if col in string_features:
          data[col] = data[col].apply(lambda x: 'rare' if x in counts else x)
        else:
          data[col] = data[col].apply(lambda x: 6 if x in counts else x) #Random value for 'rare' instance (there is no other 6)
        break
      idx = idx-1

  #Define time variable (Year, Month and Day are all equal for 1m)
  data['hour'] = data['hour'].apply(lambda x: x-14102100)  

  #Label Encoding  
  for feat in data_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat]) 


  data.to_csv('Avazu/avazu_prep.csv', index = False)
