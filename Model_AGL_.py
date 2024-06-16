
import torch, os, random, json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from transformers import logging
logging.set_verbosity_error()

# from utils_curriculum import CustomSampler
# from utils_contrast import scl
from utils_processor import *
from utils_model import *

# config 中已经添加路径了
from data_loader import ABSADataModule

"""
| dataset     | rest  | lap   | twi   |

| baseline    | 83.42 | 79.36 | 00.00 |
| performance | 82.08 | 80.31 | 75.34 | -> deberta-base

"""
baselines = {
    'base': {'rest': 0.805, 'lap': 0.79, 'twi': 0.75}, 
    'large': {'rest': 0.84, 'lap': 0.82, 'twi': 0.76},
}

class ABSADataset_AKG(ABSADataModule):
    def static_aspects(self, stage='train'):
        if stage == 'train': 
            aspects = {'term': {}, 'other': {}, 'train': {}, 'ai2a': {}, 'si2a': {}}
        else: aspects = self.aspects

        description = json.load(open(f"{self.data_dir}aspects.json", 'r'))
        for sample in self.datas[stage]:
            term = sample['aspect']
            if term not in aspects['term']: # 只统计一次 aspect
                aspects['term'][term] = {'ai': len(aspects['term']), 'desc': description[term]['description']}
                if stage!='train': aspects['other'][term] = {'desc': description[term]['description']}
                else: aspects['train'][term] = {'ai': len(aspects['train']), 'si': [], 'desc': description[term]['description']}
                aspects['ai2a'][len(aspects['term'])-1] = term

            sample['asp_ai'] = aspects['term'][term]['ai']

            if stage=='train': aspects['train'][term]['si'].append(sample['index'])
            if stage=='train': aspects['si2a'][sample['index']] = sample['aspect']
        
        return aspects
         
    def build_akg_matrix_by_sbert(self, embed_model, stage='train'):
        self.aspects = self.static_aspects(stage='train') # 统计 aspects 信息
        asp_train = list(self.aspects['train'].keys())
        asp_train_index = [self.aspects['term'][term]['ai'] for term in asp_train]
        assert all(i < j for i, j in zip(asp_train_index, asp_train_index[1:])) # 保证是递增的
        asp_train_embed = [embed_model.encode(info['desc']) for _,info in self.aspects['train'].items()]
        asp_train_sim_raw = util.cos_sim(asp_train_embed, asp_train_embed)
        asp_train_sim = asp_train_sim_raw-torch.eye(len(asp_train_embed))*1e8 # 减去自身
        self.aspects['sim_train_raw'] = asp_train_sim_raw
        self.aspects['sim_train'] = asp_train_sim
        self.aspects['embed_train'] = torch.tensor(asp_train_embed) # 只需要可见的 asp embedding

        # ## 过滤相似度, 并归一化
        # asp_train_sim_threshold = asp_train_sim.clone() 
        # asp_train_sim_threshold[asp_train_sim < threshold] = -1e8 ## 按 阈值 过滤
        # values, indices = torch.topk(asp_train_sim, topk, dim=1)
        # asp_train_sim_topk = torch.full_like(asp_train_sim, -1e8).scatter(1, indices, values) ## 按 topk 过滤
        # asp_train_sim_norm = F.softmax(asp_train_sim_topk, dim=1) ## 相似度归一化

        # for i,asp in enumerate(asp_train):
        #     asp_link = [asp_train[j] for j,v in enumerate(asp_train_sim_norm[i]) if v>-1]
        #     diss = [editdistance.eval(asp, tmp) for tmp in asp_link if tmp!=asp]
        #     self.aspects['term'][asp]['beta'] = min(diss)/max(diss) if diss else 0

        # ## 添加相似 aspect 信息, 用于 curriculum learning
        # for cur_asp, cur_asp_info in self.aspects['train'].items():
        #     cur_asp_sim = asp_train_sim[cur_asp_info['ai']].detach().cpu().numpy()    
        #     # cur_asp_sim_sort_index = sorted(range(len(cur_asp_sim)), key=cur_asp_sim.__getitem__, reverse=True)
        #     cur_asp_sim_sort_index = np.argsort(cur_asp_sim)[::-1]
        #     cur_asp_info['sim'] = cur_asp_sim[cur_asp_sim_sort_index]
        #     cur_asp_info['sim_asps'] = [self.aspects['ai2a'][ai] for ai in cur_asp_sim_sort_index]

        self.aspects = self.static_aspects(stage='valid') # 统计 aspects 信息
        asp_other = list(self.aspects['other'].keys())
        asp_other_index = [self.aspects['term'][term]['ai'] for term in asp_other]
        assert all(i < j for i, j in zip(asp_other_index, asp_other_index[1:])) # 保证是递增的
        asp_other_embed = [embed_model.encode(info['desc']) for _,info in self.aspects['other'].items()]
        asp_other_sim = util.cos_sim(asp_other_embed, self.aspects['embed_train']) 
        self.aspects['sim_other'] = asp_other_sim


        # ## 过滤相似度, 并归一化
        # asp_other_sim_threshold = asp_other_sim.clone() 
        # asp_other_sim_threshold[asp_other_sim < threshold] = -1e8 ## 按 阈值 过滤
        # values, indices = torch.topk(asp_other_sim, topk, dim=1)
        # asp_other_sim_topk = torch.full_like(asp_other_sim, -1e8).scatter(1, indices, values) ## 按 topk 过滤
        # asp_other_sim_norm = F.softmax(asp_other_sim_topk, dim=1) ## 相似度归一化

        # for i,asp in enumerate(asp_other):
        #     asp_link = [asp_train[j] for j,v in enumerate(asp_other_sim_norm[i]) if v>-1]
        #     diss = [editdistance.eval(asp, tmp) for tmp in asp_link if tmp!=asp]
        #     self.aspects['term'][asp]['beta'] = min(diss)/max(diss)

        # ## for train + other
        # asp_sim_add_other = torch.cat([self.aspects['sim_train'], asp_other_sim_norm])
        # self.aspects['sim_add'] = asp_sim_add_other

        for stage, samples in self.datas.items():
            for sample in samples:
                if stage != 'train': sample['alpha'] = 0
                else: sample['alpha'] = len(self.aspects['train'][sample['aspect']]['si'])
                # sample['beta'] = self.aspects['term'][sample['aspect']]['beta']
                    
    def get_embed(self, embed_model, inputs, batch_size=32):
        embed, ind = [], 0
        while ind < len(inputs):
            tmp = {
                'input_ids': pad_sequence([inp['input_ids'][0] for inp in inputs[ind:ind+batch_size]], 
                 batch_first=True, padding_value=self.tokenizer.pad_token_id).to(embed_model.plm_model.device),
                'attention_mask': pad_sequence([inp['attention_mask'][0] for inp in inputs[ind:ind+batch_size]], 
                 batch_first=True, padding_value=self.tokenizer.pad_token_id).to(embed_model.plm_model.device)
            }
            with torch.no_grad(): embed.append(embed_model.encode(tmp, ['cls'])['cls'])
            ind += batch_size

        return torch.cat(embed)

    def build_akg_matrix_by_self(self, embed_model, threshold=0.0, stage='train'):
        self.aspects = self.static_aspects(stage=stage) # 统计 aspects 信息
        if stage == 'train':
            asp_train = list(self.aspects['train'].keys())
            asp_train_inputs = [self.tokenizer.encode_plus(term, return_tensors='pt') for term in asp_train]
            asp_train_embed = self.get_embed(embed_model, asp_train_inputs)
            asp_train_sim = util.cos_sim(asp_train_embed, asp_train_embed) 
            threshold_mask = asp_train_sim < threshold
            asp_train_sim_norm = (asp_train_sim - threshold_mask*1e8) # 保留自身吧

            self.aspects['sim_train_raw'] = asp_train_sim
            self.aspects['sim_train'] = F.softmax(asp_train_sim_norm, dim=1)
            self.aspects['embed_train'] = torch.tensor(asp_train_embed) # 只需要可见的 asp embedding

            indices = self.get_indices()
            return indices
        else:
            asp_other = self.aspects['other']
            asp_other_inputs = [self.tokenizer.encode_plus(term, return_tensors='pt') for term in asp_other]
            asp_other_embed = self.get_embed(embed_model, asp_other_inputs)
            asp_other_sim = util.cos_sim(asp_other_embed, self.aspects['embed_train']) 
            threshold_mask = asp_other_sim < threshold
            asp_other_sim_norm = (asp_other_sim - threshold_mask*1e8)
            
            ## for train + other
            asp_sim_add_other = torch.cat([self.aspects['sim_train'], F.softmax(asp_other_sim_norm)])
            self.aspects['sim_add'] = asp_sim_add_other
    
    def get_indices(self):
        asp_train = self.aspects['train']
        asp_train_terms = list(asp_train.keys())
        indices, batch_size, sim = [], self.batch_size, self.aspects['sim_train_raw']
        self.batchs = {'ids': [], 'sim': []}
        while len(set(indices)) not in [len(self.datas['train']), len(self.datas['train'])-1]:
            tar_asp = random.choice(asp_train_terms)
            tar_asp_sim = asp_train[tar_asp]['sim']
            tar_asp_sim_asps = asp_train[tar_asp]['sim_asps']
            if (len(self.datas['train'])-len(indices))>=batch_size:
                candidate_num = batch_size//2  
            else:
                candidate_num = (len(self.datas['train'])-len(set(indices)))//2
            
            pos_candidates, neg_candidates, pos_tmp, neg_tmp = [], [], [], []
            for i in range(len(tar_asp_sim_asps)):
                for v in asp_train[tar_asp_sim_asps[i]]['si']: 
                    if len(pos_candidates)<candidate_num and v not in indices and v not in neg_candidates: 
                        pos_candidates.append(v); pos_tmp.append(tar_asp_sim[i])
                for v in asp_train[tar_asp_sim_asps[-i-1]]['si']:
                    if len(neg_candidates)<candidate_num and v not in indices and v not in pos_candidates:
                        neg_candidates.append(v); neg_tmp.append(tar_asp_sim[-i-1])

                if len(pos_candidates) >= candidate_num and len(neg_candidates) >= candidate_num: break
            
            # random.shuffle(pos_candidates); random.shuffle(neg_candidates)
            batch_tmp = pos_candidates[0:batch_size//2]+neg_candidates[0:batch_size//2]
            batch_sim_tmp = pos_tmp[0:batch_size//2]+neg_tmp[0:batch_size//2]
            if len(set(batch_tmp)-set(indices)) > 0:
                indices.extend(batch_tmp)
                self.batchs['ids'].append(batch_tmp); self.batchs['sim'].append(batch_sim_tmp)
        
        return indices

    def collate_fn(self, samples):
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        inputs['alpha'] = torch.tensor([1/s['alpha'] if s['alpha'] else 0 for s in samples])
        # inputs['beta'] = torch.tensor([1-s['beta'] for s in samples])
        # assert inputs['beta'].min() >= 0 and inputs['beta'].max() <= 1
        return inputs

def config_for_model(args, scale='base'):
    scale = args.model['scale']
    # if scale=='large': args.model['plm'] = args.file['plm_dir'] + 'bert-large'
    # else: args.model['plm'] = args.file['plm_dir'] + 'bert-base-uncased'
    args.model['plm'] = args.file['plm_dir'] + f'deberta-{scale}'

    args.model['data_dir'] = f"dataset/{args.train['tasks'][-1]}/"
    if not os.path.exists(args.model['data_dir']): os.makedirs(args.model['data_dir']) # 创建路径
    args.model['data'] = args.model['data_dir']+f"{args.model['name']}.{scale}"
    args.model['baseline'] = 0 # baselines[scale][args.train['data']]

    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['Adam', 'linear']
    # args.model['optim_sched'] = ['SGD', 'linear']
    
    args.model['epoch_every'] = False # 每个 epoch 前处理
    args.model['epoch_before'] = False # epoch 前处理
    return args

def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    
    ## 2. 导入数据
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        # paraser data
        data_dir = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
        dataset = ABSADataset_AKG(data_dir,  args.train['batch_size'], num_workers=0)   
        # tokenizer data
        tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
        dataset.setup(tokenizer)
        # aspect graph
        sbert_model = 'all-distilroberta-v1' if args.model['scale']=='base' else 'all-roberta-large-v1'
        sbert = SentenceTransformer(f"{args.file['plm_dir']}/sbert/{sbert_model}")
        dataset.build_akg_matrix_by_sbert(sbert)
        torch.save(dataset, data_path)

    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id,
        'attention_mask': 0,
        'token_type_ids': 0, 
        'asp_ai': -1, # aspect index
        'label': -1,
    }

    ## 3. 导入模型
    model = AKG(
        args=args,
        dataset=dataset,
    )
    return model, dataset
   

class AKG(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super().__init__() 
        self.args = args
        self.dataset = dataset
        self.n_class = dataset.num_classes

        if args.model['use_adapter']:
            from utils_adapter import DebertaAdapters
            self.plm_model = DebertaAdapters.from_pretrained(plm if plm is not None else args.model['plm'])
            for n, p in self.plm_model.named_parameters():
                if 'adapter' in n: p.requires_grad=True
                else: p.requires_grad=False
        else: self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_pooler = PoolerAll(self.plm_model.config)    
        self.hidden_size = self.plm_model.config.hidden_size

        self.classifier = nn.Linear(self.hidden_size, self.n_class)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_ce = CrossEntropyLoss() 

        aspects = self.dataset.aspects
        self.asp_embedding = torch.tensor(aspects['embed_train'])
        asp_num, vec_d = self.asp_embedding.shape
        if vec_d != self.hidden_size:
            self.asp_embedding = torch.cat((self.asp_embedding, torch.zeros(asp_num, self.hidden_size-vec_d)), dim=1)
        
        asp_sim_add = torch.cat([aspects['sim_train'], aspects['sim_other']])
        self.asp_similarity = F.softmax(asp_sim_add, dim=1) ## 相似度归一化

        # asp_sim_add[asp_sim_add < 0] = 0 
        # self.asp_similarity = asp_sim_add/asp_sim_add.sum(dim=-1).unsqueeze(dim=-1) ## 相似度归一化
        
        ## 过滤相似度, 并归一化
        # asp_sim_threshold = asp_sim_add.clone() 
        # asp_sim_threshold[asp_sim_threshold < args.model['threshold']] = -1e8 ## 按 阈值 过滤
        # values, indices = torch.topk(asp_sim_add, args.model['topk'], dim=1)
        # asp_sim_add_topk = torch.full_like(asp_sim_add, -1e8).scatter(1, indices, values) ## 按 topk 过滤
        # self.asp_similarity = F.softmax(asp_sim_add_topk, dim=1) ## 相似度归一化

    def epoch_deal(self, stage='train'):
        # early_threshold = self.cur_epoch - self.metrics.valid['epoch']
        # if self.cur_epoch and early_threshold <= self.args.train['early_stop']//2: 
        #     return None
        if stage == 'train':
            indices = self.dataset.get_indices()
            # 课程顺序 (indices 是batch的list)
            self.dataset.loader['train'] = DataLoader(
                self.dataset.datas['train'], 
                batch_size=self.args.train['batch_size'], 
                sampler=CustomSampler(indices),
                num_workers=0,
                collate_fn=self.dataset.collate_fn,
                )

    def encode(self, inputs, methods=['cls', 'asp', 'all']):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        plm_out = self.plm_model(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
            )

        hidden_states = self.plm_pooler(plm_out.last_hidden_state)
        #hidden_states = plm_out.last_hidden_state

        outputs = {}
        for method in methods:
            if method == 'cls': outputs['cls'] = hidden_states[:,0]
            if method == 'all': outputs['all'] = hidden_states
            if method == 'asp':
                token_type_ids = inputs['token_type_ids'] # token_type_ids 的平均值
                outputs['asp'] = torch.stack([torch.mean(hidden_states[bi][tmp.bool()], dim=0) for bi, tmp in enumerate(token_type_ids)])

        return outputs

    def forward(self, inputs, stage='train'):
        outputs = self.encode(inputs, methods=['cls', 'asp'])

        # aggregation features
        asps_ai = inputs['asp_ai'].detach().cpu() # 提取 asp 索引
        asps_ai_sim = self.asp_similarity[asps_ai] # 当前asp 与 训练集asps 的相似度
        asps_agg_features = torch.matmul(asps_ai_sim, self.asp_embedding).type_as(outputs['asp']) # 提取 asp 聚合特征
        
        # beta = inputs['beta'].unsqueeze(-1)
        asps_features = (outputs['asp'] + asps_agg_features) / 2 # 增强 asp 特征
        features = self.dropout((outputs['cls'] + asps_features) / 1) # 增强情感特征: cls+asp
        #features = self.dropout(outputs['cls'])

        # ## 加了这个效果更差
        # logits_asp = self.classifier(outputs['asp'])
        # neutral_label = torch.ones_like(inputs['label'])*self.dataset.tokenizer_['labels']['l2i']['neutral']
        # loss_asp = self.loss_ce(logits_asp, neutral_label)

        logits = self.classifier(features)
        loss = self.loss_ce(logits, inputs['label'])
        # loss = loss + loss_asp*self.args.model['loss_asp']
        
        # aspect graph learning
        if stage == 'train':
            ## 1. alpha for aggretation
            alphas = 1-inputs['alpha'].cpu() #/self.args.train['epochs'] # 根据样本频次自适应
            # alphas[alphas < 0.0] = 0.0
            for i,idx in enumerate(asps_ai):
                self.asp_embedding[idx] = self.asp_embedding[idx]*alphas[i] + outputs['asp'][i].detach().cpu()*(1-alphas[i])
            # loss_agg = F.mse_loss(outputs['asp'], asps_agg_features) # 重构（聚合）损失
            loss_agg = F.l1_loss(outputs['asp'], asps_agg_features) # 让聚合后的 asp 接近真实 asp
            
            ## 2. beta for curriculum contrastive learning
            sim_train_raw = self.dataset.aspects['sim_train']
            batch_sim = torch.tensor([[sim_train_raw[ai][_ai] for _ai in asps_ai] for ai in asps_ai])
            labels_asp_scl = batch_sim > 0.9
            # labels_asp_scl[torch.eye(labels_asp_scl.size(0), dtype=torch.bool)] = True
            loss_asp_scl = scl(outputs['asp'], labels_asp_scl, temp=1) # 让相似的 asp 更相似

            loss = loss + loss_agg*self.args.model['loss_agl'] + loss_asp_scl*self.args.model['loss_ccl']

        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }


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

            save_params_dict = {}
            for name, param in self.state_dict().items():
                if 'plm_model' in name:
                    if 'adapter' in name: save_params_dict[name] = param
                else: save_params_dict[name] = param
            save_path = self.args.file['cache_dir']+self.dataset.name[-1]
            save_path, seed = self.args.file['record'], self.args.train['seed']
            agl, ccl = self.args.model['loss_agl'], self.args.model['loss_ccl']
            torch.save(save_params_dict, save_path+f"/save_params_seed_{seed}_agl_{agl}_ccl_{ccl}.pth")
            torch.save(self.asp_embedding, save_path+f"/save_embedding_seed_{seed}_agl_{agl}_ccl_{ccl}.pth")

        # self.args.logger['loss'].info(f"epoch: {self.cur_epoch}, val_loss: {np.array(loss).mean()}")
        self.validation_step_outputs = [] # init record

def scl(embeddings, labels, temp=0.3):
    """
    calculate the contrastive loss (optimized)
    embedding: [bz, dim]
    label: [bz, bz] # True or False
    """
    # cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / temp
    # cosine_sim = torch.matmul(embeddings, embeddings.T) / temp
    # remove diagonal elements from matrix
    mask = ~torch.eye(cosine_sim.shape[0], dtype=bool,device=cosine_sim.device)
    dis = cosine_sim[mask].reshape(cosine_sim.shape[0], -1)
    # apply exp to elements
    dis_exp = torch.exp(dis)
    cosine_sim_exp = torch.exp(cosine_sim)
    row_sum = dis_exp.sum(dim=1) # calculate row sum
    # Pre-compute label counts
    # unique_labels, counts = labels.unique(return_counts=True)
    # label_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    # calculate contrastive loss
    contrastive_loss, contrastive_num = 0, 0
    for i,lab in enumerate(labels):
        n_i = lab.sum() # label_count[labels[i].item()] - 1
        inner_sum = torch.log(cosine_sim_exp[i][lab] / row_sum[i]).sum()
        contrastive_loss += inner_sum / (-n_i) if n_i != 0 else 0
        contrastive_num += 1 if n_i != 0 else 0

    return contrastive_loss/contrastive_num if contrastive_num else 0