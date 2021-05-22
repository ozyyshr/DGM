import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
import torch
from dgl import DGLGraph
 
gcn_msg=fn.copy_src(src="h",out="m")
gcn_reduce=fn.sum(msg="m",out="h")


class NodeApplyModule(nn.Module):
    def __init__(self,in_feats,out_feats,activation):
        super(NodeApplyModule,self).__init__()
        self.linear=nn.Linear(in_feats,out_feats)
        self.activation=activation

    def forward(self, node):
        h=self.linear(node.data["h"])
        if self.activation is not None:
            h=self.activation(h)
        return {"h": h}
 
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN,self).__init__()
        self.apply_mod=NodeApplyModule(in_feats,out_feats,activation)
    def forward(self,g,feature):
        g.ndata["h"]=feature
        g.update_all(gcn_msg,gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop("h")
 
class GCNNet(nn.Module):
    def __init__(self, hdim, nlayers=2, dropout_prob=0.1):
        super(GCNNet, self).__init__()
        # self.gcns = nn.ModuleList([GCN(hdim, hdim, F.relu) for i in range(nlayers)])
        feedfoward_input_dim, feedforward_hidden_dim, hidden_dim = hdim, hdim, hdim
        self._gcn_layers = [GCN(hdim, hdim, F.relu) for _ in range(nlayers)]
        self.nlayers = nlayers
        self.linear = nn.Linear(hdim, hdim)
        # self._feedfoward_layers = [FeedForward(feedfoward_input_dim,
        #                              activations=[F.relu,
        #                                           nn.Linear(hdim, hdim, bias=True)],
        #                              hidden_dims=[feedforward_hidden_dim, hidden_dim],
        #                              num_layers=2,
        #                              dropout=[dropout_prob, dropout_prob]) for _ in range(nlayers)]
        self._layer_norm_layers = [nn.LayerNorm(hdim) for _ in range(nlayers)]
        self._feed_forward_layer_norm_layers = [nn.LayerNorm(hdim) for _ in range(nlayers)]

        self.dropout = nn.Dropout(dropout_prob)
        self._input_dim = hdim
        self._output_dim = hdim

    def forward(self, g, features):
        output = features
        for i in range(self.nlayers):
            gcn = self._gcn_layers[i]
            # feedforward = self._feedfoward_layers[i]
            feedforward_layer_norm = self._feed_forward_layer_norm_layers[i]
            layer_norm = self._layer_norm_layers[i]
            cached_input = output
            feedforward_output = F.relu(self.linear(output))
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = gcn(g, feedforward_output)
            output = layer_norm(self.dropout(attention_output) + feedforward_output)
        return output

class GCN_layers(nn.Module):

    def __init__(self, hdim,
                 nlayers=2
                 ):
        super(GCN_layers, self).__init__()
        self.GCNNet = GCNNet(hdim, nlayers, 0.1)

    def transform_sent_rep(self, sent_rep, sent_mask, meta_field, key):
        init_graphs = self.convert_sent_tensors_to_graphs(sent_rep, sent_mask, meta_field, key)
        unpadated_graphs = []
        for g in init_graphs:
            updated_graph = self.forward(g)
            unpadated_graphs.append(updated_graph)
        recovered_sent = torch.stack(unpadated_graphs, dim=0)
        assert recovered_sent.shape == sent_rep.shape
        return recovered_sent

    def forward(self, g):
        h = g.ndata['h']
        out = self.GCNNet.forward(g, features=h)
        # return g, g.ndata['h'], hg  # g is the raw graph, h is the node rep, and hg is the mean of all h
        print(out)
        return out

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self):
        return False


class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated = True):
        
        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, self.feat_size))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))
        
        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
        
    def forward(self, g):
        
        weight = self.weight
        gate_weight = self.gate_weight
        
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm']
            
            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                gate = torch.sigmoid(gate)
                msg = msg * gate    
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated = True):
        super(RGCNModel, self).__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated
        
        # create rgcn layers
        self.build_model()
       
    def build_model(self):        
        self.layers = nn.ModuleList() 
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated = self.gated)
            self.layers.append(rgcn_layer)
    
    def forward(self, g):
        for layer in self.layers:
            layer(g)
        
        rst_hidden = []
        for sub_g in dgl.unbatch(g):
            rst_hidden.append(sub_g.ndata['h'])
        return rst_hidden