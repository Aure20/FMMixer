"""

Author:
    Aurelio Negri, negria@student.ethz.ch

"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from .fm import FM

class FMMixer(BaseModel):
    """Other Parameters.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model. 
    :param init_std: float, to use as the initialize std of embedding vector
    :param seed: integer, to use as random seed.  
    :param num_feat: integer, input dimension, i.e. number of features
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    """
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, l2_reg_linear=0.00001, l2_reg_embedding=0.00001, 
                 mlp_dropout = ([0,0],[0,0]), init_std=0.0001, num_feat = 67, emb_dim = 32, inner_dim = ([128],[128]), width = 4,
                 seed=1024, reduction_method = "mean", mlp_func = "gelu", task='binary', device='cpu', gpus=None):

        super(FMMixer, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        
        self.width = width
        self.reduction_method = reduction_method  
        self.emb_dim = emb_dim
        self.num_feat = num_feat 
        self.fm = FM() 
        if reduction_method == "concat":
            self.linear = nn.Linear(emb_dim*num_feat,1).to(device)
        else:
            self.linear = nn.Linear(emb_dim,1).to(device)
        mlp_dim = int(emb_dim/width)

        #Define activation function
        if mlp_func == "gelu":
            func = nn.GELU()

        if mlp_func == "relu":
            func = nn.ReLU()
        
        if mlp_func == "tanh":
            func = nn.Tanh()

        if mlp_func == "sigmoid":
            func = nn.Sigmoid()
        
        #Define the MLP structure
        self.mlp = [MLPMixer(mlp_dim, reduction_method, (MLPMixer.MLP(num_feat, inner_dim[0], func, mlp_dropout[0])),
        (MLPMixer.MLP(mlp_dim, inner_dim[1], func, mlp_dropout[1]))).to(device) for _ in range(width)]

        self.to(device)

    def forward(self, X): 
        #Get from X the sparse embedding and the dense_values
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        full_input = torch.cat(sparse_embedding_list, dim=1) #[batch_size, num_feat, emb_dim]
    
        #FM linear part, applies embedding of dimension one as well 
        logit = self.linear_model(X)

        #Second order feature interaction
        logit += self.fm(full_input) 

        #Split the input tensor along the emb_dim dimension and pass all the tensors to the MLP Mixer
        if self.width > 1:
            input_list = torch.tensor_split(full_input, self.width, dim = 2)  
            out_list = []
            for i in range(self.width):
                out_list.append(self.mlp[i](input_list[i]))
            if self.reduction_method == "concat":
                mlp_final = torch.cat(out_list, dim = 2)
                mlp_final = mlp_final.view(-1,self.emb_dim*self.num_feat)
            else:
                mlp_final = torch.cat(out_list, dim = 1)
        else:
            mlp_final = self.mlp[0](full_input)
            if self.reduction_method == "concat":
                mlp_final = mlp_final.view(-1,self.emb_dim*self.num_feat)  
                      
        #Apply last connected layer
        logit += self.linear(mlp_final)

        #Apply sigmoid
        y_pred = self.out(logit) 
        
        return y_pred

    
class MLPMixer(nn.Module):
    def __init__(self, dim, reduction_method, mlp_one, mlp_two):
        super().__init__() 
        self.norm = nn.LayerNorm(dim)
        self.mlp_one = mlp_one
        self.mlp_two = mlp_two
        self.reduction_method = reduction_method
    
    def forward(self, x):#Pass all the input as a tensor
        #First MLP part
        mlp_in = self.norm(x)
        mlp_in = torch.transpose(mlp_in,1,2) #[batch_size, emb_dim/width, num_feat]
        mlp_out = self.mlp_one(mlp_in)
        mlp_out = torch.transpose(mlp_out,1,2)#[batch_size, num_feat, emb_dim/width]

        #Skip connection one
        skip = mlp_out.add(x)

        #Second MLP part
        mlp_in = self.norm(skip)
        mlp_out = self.mlp_two(mlp_in)

        #Skip Connection two
        mlp_final = mlp_out.add(skip)

        #Average over the features dimension
        if self.reduction_method == "mean":
            mlp_final = torch.mean(mlp_final,1) 
        if self.reduction_method == "max":
            mlp_final = torch.amax(mlp_final,1) 
        if self.reduction_method == "min":
            mlp_final = torch.amin(mlp_final,1) 

        return mlp_final
    
        #Defines the MLPs in the MLP mixer (Can be an arbitrary number of layers)
    def MLP(input_dim, inner_dim, func, dropout):

        if len(inner_dim) == 0:
            return nn.Sequential(nn.Linear(input_dim,input_dim), func, nn.Dropout(dropout[0]))

        mlps = []
        for i in range(len(inner_dim)):
            if i == 0:
                mlps.append(nn.Linear(input_dim, inner_dim[i]))
                mlps.append(func)
                mlps.append(nn.Dropout(dropout[0]))
            else:
                mlps.append(nn.Linear(inner_dim[i-1], inner_dim[i]))
                mlps.append(func)
                mlps.append(nn.Dropout(dropout[i]))
        #Last MLP layer
        mlps.append(nn.Linear(inner_dim[i], input_dim))
        mlps.append(nn.Dropout(dropout[i+1]))
        return nn.Sequential(*mlps)