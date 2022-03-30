# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from FMMixer.models import *
from FMMixer.inputs import SparseFeat, get_feature_names

from criteo_preprocess import criteo_prep

#Run this the first time to do the preprocessing, the it can be commented out
criteo_prep("Criteo/criteo_100k.txt")
print("Preprocessing done")

if __name__ == "__main__":
    cr_col = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I13','C1','C2','C3','C4','C5'
    ,'C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C23','C24','C25','C26'
    ,'Tree1','Tree2','Tree3','Tree4','Tree5','Tree6','Tree7','Tree8','Tree9','Tree10','Tree11','Tree12','Tree13','Tree14','Tree15'
    ,'Tree16','Tree17','Tree18','Tree19','Tree20','Tree21','Tree22','Tree23','Tree24','Tree25','Tree26','Tree27','Tree28','Tree29','Tree30']
    data = pd.read_csv('Criteo/criteo_prep.csv',  usecols = cr_col, delimiter=',', engine = 'python', nrows = 10000) 
    cr_col.remove('label')
    target = ['label']

    #Use label encoder to avoid for count values to go over embed_dim
    for feat in cr_col:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])  

    def train(data, l2_reg_embedding, l2_reg_linear, batch_size, embedding_dim, mlp_dropout, 
    inner_dim, reduction_method, mlp_func, width):

        assert embedding_dim % width == 0, "Emb_dim has to be multiple of width"

        # Count #unique features for each sparse field, assigns embedding dimension and record dense feature field name 
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim = embedding_dim)
                                for feat in cr_col]
        
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = train_test_split(data, test_size=0.2, random_state=2022)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = FMMixer(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
        l2_reg_linear=l2_reg_linear, num_feat = len(cr_col), l2_reg_embedding=l2_reg_embedding, device=device, emb_dim = embedding_dim, 
        mlp_dropout=mlp_dropout,  inner_dim=inner_dim,reduction_method = reduction_method, mlp_func = mlp_func, width = width)

        model.compile("adagrad", "binary_crossentropy",metrics=["acc", "binary_crossentropy", "auc"],)      
        
        history = model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=5, verbose=2,
                            validation_split=0.1)
        pred_ans = model.predict(test_model_input, batch_size)
        print("")
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))   
        print("test Accuracy", round(accuracy_score(test[target].values, np.around(pred_ans)), 4))
    
    """Train Parameters:
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector 
    :param batch_size: integer, defines the batch size
    :param embedding_dim: integer, defines the embedding dimension
    :param mlp_dropout: tuple of arrays, defines the dropout for each fully connected layer (length of the arrays has 
    to be +1 compared to the inner_dim ones)
    :param inner_dim: tuple of integer arrays, defines the inner dimensions of the two mlps (can also be of size 0)
    :param reduction_method: string ("concat", "max", "min" or "mean"), defines how we reduce
     the output of the MLP (mean, max or min)
    :param mlp_func: string ("relu, "gelu" or "tanh"), Activation function used in the MLP
    :param width: integer, the width of the Mixer part
    """
    train(data,1e-4,1e-4,64,128,([0, 0], [0, 0]),([128], [128]),"concat","gelu",4)  