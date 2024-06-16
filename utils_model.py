import torch, math, json
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl 
from utils_processor import *


def totally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print(n_params)
    
    return n_params

def init_weight(model, method='xavier_uniform_'):
    if method == 'xavier_uniform_': fc = torch.nn.init.xavier_uniform_
    if method == 'xavier_normal_':  fc = torch.nn.init.xavier_normal_
    if method == 'orthogonal_':     fc = torch.nn.init.orthogonal_

    ## 非 plm 模型参数初始化
    for name, param in model.named_parameters():
        if 'plm' not in name: # 跳过 plm 模型参数
            if param.requires_grad:
                if len(param.shape) > 1: fc(param) # 参数维度大于 1
                else: 
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)

def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        params_all, params_train = 0, 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel

            params_all += num_params
            if param.requires_grad: params_train += num_params
        print(f"trainable params: {params_train} || all params: {params_all} || trainable%: {100*params_train/params_all}")


class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class ModelForClassification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.cur_epoch = 0

    def forward(self, inputs, stage='train'):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = get_optimizer(self)
        return {'optimizer': optimizer}

        scheduler = get_scheduler(self.args, optimizer, len(self.dataset.train_dataloader()))
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
    
        # weight_decay = 1e-6  # l2正则化系数
        # # 假如有两个网络，一个encoder一个decoder
        # optimizer = optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=learning_rate, weight_decay=weight_decay)
        # # 同样，如果只有一个网络结构，就可以更直接了
        # optimizer = optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # # 我这里设置2000个epoch后学习率变为原来的0.5，之后不再改变
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.5)
        # optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        # return optim_dict
    
        # return torch.optim.AdamW(self.parameters(), lr=self.args.train['learning_rate'])

    def training_step(self, batch, batch_idx):
        output, cur_e = self(batch, stage='train'), self.cur_epoch,
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }
    
    def on_train_epoch_end(self):
        outputs, metrics_tr = self.training_step_outputs, self.metrics.train
        loss = [ob['loss'].item() for ob in outputs]
        # metrics calculation
        metrics = self.metrics.get_metric(outputs, self.args.train['tasks'][0], stage='train')
        metrics['loss'] = round(np.array(loss).mean(), 4) 

        metrics_tr.update(metrics)
        metrics_tr['epoch'] = self.cur_epoch
        
        # self.args.logger['loss'].info(f"epoch: {self.cur_epoch}, train_loss: {np.array(loss).mean()}")
        self.training_step_outputs = [] # init record
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_tr.items()})

        self.args.logger['process'].info(f"train_eval: {describe}")
    
    def validation_step(self, batch, batch_idx):
        output = self(batch, stage='valid')
        self.validation_step_outputs.append(output)

        return output

    def on_validation_end(self):
        outputs, metrics_vl = self.validation_step_outputs, self.metrics.valid
        loss = [ob['loss'].item() for ob in outputs]
        # metrics calculation
        metrics = self.metrics.get_metric(outputs, self.args.train['tasks'][0], stage='valid')
        metrics['loss'] = round(np.array(loss).mean(), 4) 
        ## update best model
        mark, self.valid_update = self.dataset.metric, False
        if metrics[mark] > metrics_vl[mark]: # bigger is better
            metrics_vl.update(metrics)
            metrics_vl['epoch'] = self.cur_epoch
            describe = json.dumps({k: round(float(v),4) for k,v in metrics_vl.items()})
            self.args.logger['process'].info(f"valid: {describe}")
            self.valid_update = True # execute test

        # self.args.logger['loss'].info(f"epoch: {self.cur_epoch}, val_loss: {np.array(loss).mean()}")
        self.validation_step_outputs = [] # init record

    def test_step(self, batch, batch_idx):
        output = self(batch, stage='test')
        self.test_step_outputs.append(output)

    def on_test_end(self):
        outputs, metrics_te = self.test_step_outputs, self.metrics.test
        loss = [ob['loss'].item() for ob in outputs]
        # metrics calculation
        metrics = self.metrics.get_metric(outputs, self.args.train['tasks'][0], stage='test')
        metrics['loss'] = round(np.array(loss).mean(), 4) 

        metrics_te.update(metrics)
        metrics_te['epoch'] = self.cur_epoch
        
        # self.args.logger['loss'].info(f"epoch: {metrics_te['epoch']}, test_loss: {np.array(loss).mean()}")
        self.test_step_outputs = []
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_te.items()})
        self.args.logger['process'].info(f"test: {describe}")