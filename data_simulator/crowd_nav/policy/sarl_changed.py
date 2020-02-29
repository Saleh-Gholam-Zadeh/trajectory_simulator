print('sarl_changed_ imported')
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL

import numpy
import sys
import time
#print('sarl_changed_ imported')





class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    print('-------------------------debug-sarl_changed.py--line 19:--------------------')
    print('Encoder is instanciated')
    def __init__(
        self, embedding_dim=64, h_dim=64 ,  num_layers=1,dropout=0.0,self_state_dim=6):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.self_state_dim = self_state_dim

        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
                                                ) #relu nazashte tu code sgan  vali   dakhelepaper relu gozashte
        self.lstm = nn.LSTM( embedding_dim, h_dim, num_layers, dropout=dropout ,batch_first=False) #(seq_len , batch ,#feature)



    def forward(self, state):
        """
         First transform the world coordinates to self-centric coordinates and then do forward computation

         :param state: tensor of shape (batch_size, # of humans, length of a joint state)
         :return:
         """
        # Encode observed Trajectory


        size = state.shape
        #print('size= ',size)
        seq_len = size[0]
        lstm_batch_size = size[1] #=5   add +1 if you want to add robot which is not usefull in robot centric formulasion
        # print('---------debug -encodder')
        # print('state.shape',state.shape) #[37,5,13]
        # print('self.state_dim.shape',self.self_state_dim) #6

        self_state = state[:, 0, :self.self_state_dim] #[37 ,6]
        self_state = self_state.unsqueeze(1)#[37,1,6]
        #print(self_state.shape , 'self_state.shape')

        human_state = state[:, :, self.self_state_dim:] #[37,5,7]

        h0 = torch.zeros(1, lstm_batch_size, self.h_dim)
        c0 = torch.zeros(1, lstm_batch_size, self.h_dim)

        inp = human_state[:,:,0:2]
        #inp = torch.cat([torch.zeros([seq_len,size[1],2]), human_state[:,:,0:2]], dim=1) #[37,1,2] + [37,5,2]   robot ra nabayad ezafe konim
        #print('inp.shape = ', inp.shape)
        embedding = self.spatial_embedding(inp.view(-1, 2))

        embedding = embedding.view(seq_len,lstm_batch_size,self.embedding_dim)
        #print('embedding.shape = ',embedding.shape)#[37,5,64]

        output, (hn, cn) = self.lstm(embedding, (h0, c0))
        #print('output.shape',output.shape) # (37,5,64=h_dim)
        #print('hn.shape = ',hn.shape)#[1,5,64=h_dim]
        hn = hn.squeeze(0)
        #print('hn.shape = ', hn.shape)#[6,64=h_dim]

        #print('Encoder output.shape',output.shape)#[37,5,64]
        return output


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in SOcial Gan paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,self_state_dim=6,
        activation='relu', batch_norm=True, dropout=0.0):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.self_state_dim = self_state_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Sequential (                    #CHECK KON RELU MIKHAD YA NA?
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
                                            )

        self.mlp_pre_pool =mlp(mlp_pre_pool_dims[0],mlp_pre_pool_dims[1:] ,last_relu=True)

    def forward(self, state,hidden):
        """
         First transform the world coordinates to self-centric coordinates and then do forward computation

         :param state: tensor of shape (batch_size, # of humans, length of a joint state)
         :return:
         """
        size = state.shape
        #print('size= ',size)
        seq_len = size[0]
        pooling_batch_size = size[1] #5 human
        #print('---------debug -PM')
        #print('hidden.shape = ',hidden.shape)

        self_state = state[:, 0, :self.self_state_dim] #[37 ,6]
        self_state = self_state.unsqueeze(1)#[37,1,6]

        human_state = state[:, :, self.self_state_dim:] #[37,5,7]
        #self_hidden = hidden[:,0,:]    # [37,1,64 =h_dim]
        human_hidden = hidden[:,:,:]  # [37,5,64=h_dim]

        inp1 =  human_state[:,:,0:2] #all [px,py]  [37,5,2]
        #print('inp.shape = ', inp1.shape)
        embedding = self.spatial_embedding(inp1.view(-1, 2))

        embedding = embedding.view(seq_len,pooling_batch_size,self.embedding_dim)  #[37,5,64]
        #print('embedding.shape = ',embedding.shape)#[37,5,64]

        inp2 = torch.cat([embedding,human_hidden],dim =2 )  # [37,5,56+64]
       # print('inp2.shape = ',inp2.shape)
        inp2 = inp2.view(-1,self.embedding_dim+self.h_dim)  #[-1,128]
        output2 = self.mlp_pre_pool(inp2) #[37*5 ,1024]

        inp3=output2.view(seq_len,pooling_batch_size,self.bottleneck_dim) #[37,5,1024]

        output3 = inp3.max(1)[0]  # [37,1024]
        return output3







class New_ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        #print('self_state_dim = ',self.self_state_dim)
        self.global_state_dim = mlp1_dims[-1]
        self.encoder = Encoder(embedding_dim=64 , h_dim=64,self_state_dim=self.self_state_dim)
        self.pooling =PoolHiddenNet(embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024)

        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim =1024+6 # PM_size + robot(self)_state
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None
        ####    #####

        #print('sarl.py line 31: input_dim,self_state_dim = ',input_dim,self_state_dim )
    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        # state.shape =[100(#batch) * 5(#human) *13(#state)]
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]    #self_state_dim=6, mlp1_output=[100 *6]
        #mlp1_output = self.mlp1(state.view((-1, size[2]))) #  state.view((-1, size[2])) = 500*13         mlp1_output=[500*100]     (13in-->100out)
        #mlp2_output = self.mlp2(mlp1_output)        #mlp2_output [500*50]   (100in--->50out)

        encoder_output = self.encoder(state)
        #print('------------------ debug sarl_changed forward pass: ------------------------------------')
        #print('encoder_output.shape = ' , encoder_output.shape)

        pooling_output = self.pooling(state,encoder_output)

        #print('pooling_output.shape = ',pooling_output.shape)  #[37,1024+64]
        #inp_value_net = torch.cat([pooling_output,encoder_output[:,0,:]],1)  => [PM , hei]
        #print("--debug: sarl_changed : line 193")
        #print("self_state.shape = ",self_state.shape)
        #print("pooling_output.shape = ", pooling_output.shape)

        inp_value_net = torch.cat([self_state,pooling_output],1)  #[37,6] , [37,1024]
        value = self.mlp3(inp_value_net) #joint_state =[100*56]
        #print('VALUE IS READY')
        return value


class SARL(MultiHumanRL):
    #print('---------------------')
    #print('sarl_changed_ imported')
    #print('---------------------')
    def __init__(self):
        super().__init__()
        self.name = 'SARL'
        ##
        #print('debug: SARL object is created ')
        ##

    def configure(self, config):

        # --- debug --

        # print('---------------------')
        # print('sarl_changed_ imported')
        # print('---------------------')
        # print('------debug:sarl_changed.py')
        # print('------debug:configure')
        # print('class SARL(MultiHumanRL) Configure is called')


        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')    # default =false
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = New_ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om: # false default
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

        #print('------EndDebug:configure')

    def get_attention_weights(self):

        # print('---------------------')
        # print('get_attention_weight')
        # print('---------------------')
        return self.model.attention_weights
