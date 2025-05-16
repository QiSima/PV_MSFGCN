import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)
    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha,beta):
        super(mixprop, self).__init__()
        self.nconv1 = nconv()  
        self.nconv2 = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha  
        self.beta  = beta   
    def forward(self, x, adjs):
        matrix = []
        for adj in adjs:
            adj = adj + torch.eye(adj.size(0)).to(x.device)
            d = adj.sum(1)
            h = x
            out = [h]
            a = adj / d.view(-1, 1)
            matrix.append(a)
        
        if len(matrix) > 0:
            for i in range(self.gdep):
                h = self.alpha*x + self.beta*self.nconv1(h,matrix[0])+ (1-self.alpha-self.beta)*self.nconv2(h,matrix[1])
                out.append(h)
        else:
            for i in range(self.gdep):
                h = self.alpha*x + (1-self.alpha)*self.nconv1(h,matrix[0])
                out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class GraphBlock_MIX(nn.Module):
    def __init__(self, c_out , conv_channel, skip_channel, gcn_depth , dropout, 
                  propalpha , propbeta, node_dim):
        super(GraphBlock_MIX, self).__init__()
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha, propbeta)
        self.gelu = nn.GELU()

    def forward(self, x, pre_adj):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.transpose(1, 3) 
        out = self.gelu(self.gconv1(out , [adp,pre_adj])) 
        return out.transpose(1, 3)

class ScaleMixing(nn.Module):
    """
    Bidirectioncal mixing season pattern: up,down
    """

    def __init__(self, configs):
        super(ScaleMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )
        self.up_sampling_layers = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                            configs.seq_len // (configs.down_sampling_window ** i),
                        ),
                        nn.GELU(),
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len // (configs.down_sampling_window ** i),
                        ),
                    )
                    for i in reversed(range(configs.down_sampling_layers))
                ])

    def forward(self, enc_list):
        # mixing high->low
        out_high = enc_list[0]
        out_low = enc_list[1]
        out_enc_list_2 = [out_high.permute(0, 2, 1)]
        for i in range(len(enc_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(enc_list) - 1:
                out_low = enc_list[i + 2]
            out_enc_list_2.append(out_high.permute(0, 2, 1))

        # mixing low->high
        enc_list_reverse = enc_list.copy()
        enc_list_reverse.reverse()
        out_low = enc_list_reverse[0]
        out_high = enc_list_reverse[1]
        out_enc_list_1 = [out_low.permute(0, 2, 1)]
        for i in range(len(enc_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(enc_list_reverse) - 1:
                out_high = enc_list_reverse[i + 2]
            out_enc_list_1.append(out_low.permute(0, 2, 1))
        out_enc_list_1.reverse()
                
        return out_enc_list_1,out_enc_list_2

class DecomposableBiMixing(nn.Module):
    def __init__(self, configs):
        super(DecomposableBiMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        else:
            raise ValueError('decompsition is error')

        self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.conv_channel, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.bi_mixing_multi_scale_season = ScaleMixing(configs)
        # Mxing trend
        self.bi_mixing_multi_scale_trend = ScaleMixing(configs)

        self.out_cross_layer1 = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        self.out_cross_layer2 = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.conv_channel),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bi season mixing
        out_season_list_1,out_season_list_2 = self.bi_mixing_multi_scale_season(season_list)
        # bi trend mixing
        out_trend_list_1,out_trend_list_2 = self.bi_mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season_1, out_season_2,out_trend_1, out_trend_2,length in zip(x_list, out_season_list_1,out_season_list_2,out_trend_list_1,out_trend_list_2,length_list):
            out1 = self.out_cross_layer1(out_season_1+out_season_2)
            out2 = self.out_cross_layer1(out_trend_1+out_trend_2)
            out = ori + self.out_cross_layer2(out1 + out2)
            out_list.append(out[:, :length, :])
        return out_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.dbm_blocks = nn.ModuleList([DecomposableBiMixing(configs)
                                         for _ in range(configs.e_layers)])
        
        self.gconv = nn.ModuleList()
        for i in range(configs.down_sampling_layers+1):  
            self.gconv.append(
                GraphBlock_MIX(configs.c_out , configs.d_model,  configs.conv_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha,configs.propbeta, configs.node_dim))

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
        self.projection_layer = nn.Linear(
            configs.conv_channel, 1, bias=True)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return [x_enc], [x_mark_enc]
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Scale construction and input embedding
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        enc_out_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                # embedding based on x_mark ——> B*N,T,F ——> B*N,T,d_model 
                enc_out = self.enc_embedding(x, x_mark)  
                enc_out = enc_out.reshape(B,N,T,self.configs.d_model).transpose(1, 2)   
                # Adaptive multi-graph convolution for spatial feature extraction
                enc_out = self.gconv[i](enc_out,self.pre_adj).transpose(1, 2).reshape(B*N, T ,-1)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                # embedding ——> B*N,T,F ——> B*N,T,d_model 
                enc_out = self.enc_embedding(x, None)  
                enc_out = enc_out.reshape(B,N,T,self.configs.d_model).transpose(1, 2)
                # Adaptive multi-graph convolution for spatial feature extraction
                enc_out = self.gconv[i](enc_out,self.pre_adj).transpose(1, 2).reshape(B*N, T ,-1)
                enc_out_list.append(enc_out)

        # Decomposed bidirectional fusion for temporal feature extraction
        for i in range(self.layer):
            enc_out_list = self.dbm_blocks[i](enc_out_list)

        # Multi-scale fusion for multi-sites PV power prediction
        dec_out_list = self.future_multi_predictors(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_predictors(self, B, enc_out_list, x_list):
        dec_out_list = []

        x_list = x_list[0]
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                0, 2, 1)  # align temporal dimension
            dec_out = self.projection_layer(dec_out) # align features dimension
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
