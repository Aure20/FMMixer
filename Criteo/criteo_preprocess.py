import pandas as pd 
from sklearn.preprocessing import LabelEncoder    
import math
import xgboost as xgb  

def criteo_prep(data_path):
    #Label, integer and categorical features
    cr_columns = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','C1','C2','C3','C4','C5'
    ,'C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']
    value_features = ['I' + str(i) for i in range (1,14)]
    sparse_features = ['C' + str(i) for i in range (1,27)] 

    #Read data, set from_txt to true if the data is being read directly from the original criteo dataset
    data = pd.read_csv(data_path, names = cr_columns, delimiter='\t', engine = 'python')

    #Remove nan columns
    idx_len = len(data.index)*0.7
    for col in cr_columns:
        if data[col].isnull().sum() > idx_len:
            data.drop(columns = col, inplace=True)
            cr_columns.remove(col)
            if col in value_features:
                value_features.remove(col)  
            else:
                sparse_features.remove(col) 

    #Fillnan 
    data[sparse_features] = data[sparse_features].fillna('nan') 
    for feat in value_features:
        data[feat] = data[feat].fillna(round(data[feat].mean()))

    #Remove rare categories from categorical features (use dics and map to make it way faster), apply label encoder
    low_count = 10   
    for col in sparse_features: 
        lbe = LabelEncoder()
        counts = data.groupby(col).size().sort_values(ascending=False)  
        idx = len(counts)
        for val, count in counts.items():
            if count < low_count:  
                to_del = counts.tail(idx).index.tolist()
                to_keep = counts.head(len(counts) - idx).index.tolist() 
                dic_to_del = dict.fromkeys(to_del,'rare')
                dic_to_keep = dict(zip(to_keep, to_keep)) 
                dic_to_keep.update(dic_to_del)
                data[col] = data[col].map(dic_to_keep) 
                break
            idx = idx-1    
        data[col] = lbe.fit_transform(data[col])

    #Create trees
    train = xgb.DMatrix(data.drop(columns = 'label').values, label = data['label'].values)
    params = {'objective' : 'binary:logistic', 'max_depth': 7, 'eta' : 0.5}
    num_rounds = 30
    bst = xgb.train(params,train,num_rounds) 
    
    #Predict the leaf of the tree
    preds = bst.predict(train, pred_leaf = True)
    tree_col = ['Tree' + str(i) for i in range(1,num_rounds+1)]
    leaf_data = pd.DataFrame(preds,columns = tree_col)
    data = pd.concat([data,leaf_data], axis = 1)

    #Apply log
    for feat in value_features: 
        data[feat] = data[feat].apply(lambda x: math.floor(math.log(x,2)) if x > 2 else 1)

    #Get csv
    data.to_csv('Criteo\criteo_prep.csv', index = False)
            