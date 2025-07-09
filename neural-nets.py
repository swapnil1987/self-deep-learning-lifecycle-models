import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#==========================
#Evaluation
#==========================

def eval_NN(NN, NNT, x, scale_vec=None, t0=0, t=None):

    """
    Evaluate a neural network NN on input x, which is either a single time step
    of size (Nx, Ninputs) or a sequence of time steps of size (T, ..., Ninputs).

    If scale_vec is not None, scale x by scale_vec before evaluation.

    If t is not None, add time dummies to x.  If t is an integer, add a time
    dummy of size (Nx, NNT) where the t-th column is all ones.  If t is a
    sequence of integers, add a time dummy of size (T * Nx, NNT) where the
    t-th column of each block is all ones.

    Returns a tensor of size (T * Nx, Noutputs) if t is not None, and a tensor
    of size (Nx, Noutputs) if t is None.

    """
    #a. scale
    if scale_vec is not None: x = x * scale_vec
        
    
    #b. add time dummies
    if t is not None:

        #x.shape = (Nx, Ninputs)
        Nx = x.shape[0]
        time_dummies = torch.zeros((Nx, NNT), dtype = x.dtype, device = x.device)
        time_dummies[:,t] = 1
        
        x_time_dummies = torch.cat((x, time_dummies), dim = -1) #shape = (Nx, Ninputs + NNT)

        return NN(x_time_dummies)
    
    else:

        # x.shape = (T, ..., Ninputs)

        x_ = x.reshape((x.shape[0], -1, x.shape[-1])) #shape = (T, Nx, Ninputs)

        T = x_.shape[0]
        Nx = x_.shape[1]

        eyeT = torch.eye(NNT, NNT, dtype = x.dtype, device = x.device)[t0:t0 + T, :] #shape = (T, NNT)
        eyeT = eyeT.repeat_interleave(Nx, dim = 0).reshape((T, Nx, NNT)) #shape = (T, Nx, NNT)

        x_time_dummies = torch.cat((x_, eyeT), dim = -1) #shape = (T, Nx, Ninputs + NNT)
        x_time_dummies = x_time_dummies.reshape((T * Nx, -1)) #shape = (T * Nx, Ninputs + NNT)

        return NN(x_time_dummies)
    

#=============
#
#Policy
#
#=============
class Policy(nn.Module):

    def __init__(self, par, train):

        """
        __init__ method of Policy class

        Parameters
        ----------
        par: Parameters
            parameters instance
        train: Training
            training instance

        Attributes
        ----------
        Nstates: int
            number of states
        T: int
            number of time steps
        Nactions: int
            number of actions
        intermediate_activation: function
            activation function for intermediate layers
        layers: nn.ModuleList
            list of layers
        policy_activation_final: function or list of functions
            activation function for the final layer
        """
        super().__init__()

        self.Nstates = par.Nstates
        self.T = par.T
        self.Nactions = par.Nactions
        self.intermediate_activation = getattr(F, train.policy_activation_intermediate)

        self.layers = nn.ModuleList([None] * (train.Nneurons_policy.size + 1))

        # inpute layers
        self.layers[0] = nn.Linear(self.Nstates + self.T, train.Nneurons_policy[0])

        #hidden layers
        for i in range(1, len(self.layers) -1):

            self.layers[i] = nn.Linear(train.Nneurons_policy[i - 1], train.Nneurons_policy[i])

        
        #output layer
        self.layers[-1] = nn.Linear(train.Nneurons_policy[-1], self.Nactions)


        if len(train.policy_activation_final) == 1:

            if hasattr(F, train.policy_activation_final[0]):

                self.policy_activation_final = getattr(F, train.policy_activation_final[0])

            else:

                raise ValueError(f'policy_activation_final {train.policy_activation_final[0]} function not available')
            
        else:

            self.policy_activation_final = []

            for i in range(len(train.policy_activation_final)):

                if train.policy_activation_final[i] == None:

                    self.policy_activation_final.append(None)

                else:

                    if hasattr(F, train.policy_activation_final[i]):

                        self.policy_activation_final.append(getattr(F, train.policy_activation_final[i]))

                    else:

                        raise ValueError(f'policy_activation_final {train.policy_activation_final[i]} function not available')
                    
            
        
    
    def forward(self, state):

        
        """
        Forward pass of the policy function
        """
        #input layer
        s = self.intermediate_activation(self.layers[0](state))

        #hidden layers
        for i in range(1, len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))


        #output layers 
        if type(self.policy_activation_final) is not list:

            action = self.policy_activation_final(self.layers[-1](s))

        else:

            s_noact = self.layers[-1](s) #output layer without activation

            for i_a in range(self.Nactions):
                if self.policy_activation_final[i_a] is None:
                    action_i_a = s_noact[:,i_a].view(-1,1)
                else:

                    action_i_a = self.policy_activation_final[i_a](s_noact[:,i_a].view(-1,1))

                if i_a == 0:
                    action = action_i_a
                else:
                    action = torch.cat([action, action_i_a], 1)

        return action
    



def eval_policy(model, NN, states, t0=0, t=None):

    """
    Evaluate a policy function NN on input states, which is either a single time step
    of size (Nx, Ninputs) or a sequence of time steps of size (T, ..., Ninputs).

    If scale_vec is not None, scale states by scale_vec before evaluation.

    If t is not None, add time dummies to states.  If t is an integer, add a time
    dummy of size (Nx, NNT) where the t-th column is all ones.  If t is a

    t-th column of each block is all ones.

    of size (Nx, Noutputs) if t is None.

    """
    train = model.train

    if train.use_input_scaling:
        scale_vec = model.par.scale_vec_states
    else:
        scale_vec = None

    
    actions = eval_NN(NN, par.T,  states, scale_vec, t0, t)
    actions = actions.clamp(train.min_actions, train.max_actions)

    return actions




#==============
#
#Value
#
#==============
class StateValue(nn.Module):

    def __init__(self, par, train, outputs=1):

        super().__init__()

        if train.algoname == "DeepVPD" or train.algoname == "DeepVPDDC":
            self.Nstates = par.Nstates_pd
        else:
            self.Nstates = par.Nstates


        self.T = par.T
        self.intermediate_activation = getattr(F, train.value_activation_intermediate)
        self.outputs = outputs


        self.layers = nn.ModuleList()