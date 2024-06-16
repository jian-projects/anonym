import os, torch, random, json, copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
from xml.etree import ElementTree as ET

class DataLoader_ABSA(Dataset):
    def __init__(self, dataset, d_type='multi', desc='train') -> None:
        self.d_type = d_type
        self.samples = dataset.datas['data'][desc]
        self.batch_cols = dataset.batch_cols
        self.tokenizer_ = dataset.tokenizer_
        self.tokenizer = dataset.tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, output = self.samples[idx], {}
        for col, pad in self.keys.items():
            if col == 'audio':
                wav, _sr = sf.read(sample['audio'])
                output[col] = torch.tensor(wav.astype(np.float32))[0:160000]
            elif col == 'label':
                output[col] = torch.tensor(self.ltoi[sample[col]])
            else:
                output[col] = sample[col]
        return output

## absa dataset
class ABSADataset_MA(Dataset):
    def __init__(self, path, lower=False):
        self.lower = lower
        self.name = ['absa', path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        for desc in ['train', 'test']:
            self.datas['data'][desc] = self.get_dataset(path, desc)
        self.datas['data']['valid'] = self.datas['data']['test']
        self.get_tokenizer_(self.datas['data']['train'], names=['labels'])
        self.n_class = len(self.tokenizer_['labels']['ltoi'])

    def container_init(self, only='all'):
        self.info = {
            'max_seq_token_num': {}, # 句子 最长长度
            'max_asp_token_num': {}, # aspect 最长长度
            'total_samples_num': {}, # 样本数量
            'class_category': {},    # 类别统计
        }

        # 初始化数据集要保存的内容 
        self.datas = {
            'data': {},   # 解析后纯文本
            'loader': {}, # dataloader用于构建batch
        }

        # tokenizer
        self.tokenizer_ = {
            'labels': { 'ltoi': {}, 'itol': {}, 'count': {} }
        }

    def get_tokenizer_(self, samples, names):
        if not isinstance(names, list): names = [names]
        if 'polarity' in names:
            for samp in samples:
                value = samp['polarity']
                if value not in self.tokenizer_['labels']['ltoi']:
                    self.tokenizer_['labels']['ltoi'][value] = len(self.tokenizer_['labels']['ltoi'])
                    self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = value
                    self.tokenizer_['labels']['count'][value] = 1
                self.tokenizer_['labels']['count'][value] += 1

    def get_dataset(self, path, desc):
        save_path = f'{path}/{desc}.multiple.json'
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as fr:
                samples = json.load(fr)
            return samples
        else:
            raw_path = f'{path}/{desc}.raw.json'
            with open(raw_path, 'r', encoding='utf-8') as fp:
                raw_samples, samples = json.load(fp), []
            
            # nlp = spacy.load('en_core_web_trf')
            for sample in raw_samples:
                if not self.lower: tokens = sample['token']
                else: tokens = [token.lower() for token in sample['token']]                    
                assert '' not in tokens and ' ' not in tokens

                # 确保 aspect 位置正确, 且按顺序出现
                sentence, aspects = sample['sentence'], sample['aspects']
                for aspect in aspects:
                    if self.lower: aspect['term'] = [token.lower() for token in aspect['term']] 
                    assert aspect['term'] == tokens[aspect['from']: aspect['to']]
                asp_begins = [aspect['from'] for aspect in aspects]
                if not all(x<=y for x, y in zip(asp_begins, asp_begins[1:])):
                    modify_index = sorted(range(len(asp_begins)), key=lambda x:asp_begins[x])
                    aspects= [aspects[idx] for idx in modify_index]

                # sentence = ' '.join(tokens)
                # tags, deprels, heads, words = [], [], [], []
                # for word in nlp(sentence):
                #     tags.append(word.tag_)
                #     deprels.append(word.dep_)
                #     heads.append(word.head.i)
                #     words.append(str(word))
                # # head 指向 i：第i个token的父节点是第i个head对应位置的节点
                # heads = [-1 if head==i else head for i, head in enumerate(heads)]
                # assert -1 in heads
                
                # 纠正 aspect 位置
                # if len(words) != len(tokens):
                #     assert len(words) > len(tokens)
                #     aspect_token_range, new_aspect_token_range, decay = [], [], 0
                #     for aspect in aspects: aspect_token_range.extend([aspect['from'], aspect['to']])
                #     for t, token in enumerate(tokens):
                #         if t in aspect_token_range: new_aspect_token_range.append(t+decay)
                #         if token == words[t+decay]: continue
                #         if t == len(tokens)-1: continue
                #         next_token, decay_add = tokens[t+1], 1
                #         while words[t+decay+decay_add] != next_token: decay_add += 1
                #         decay += (decay_add-1)

                #     for i, val in enumerate(new_aspect_token_range):
                #         if i%2 == 1: continue
                #         aspects[i//2]['from'] = new_aspect_token_range[i]
                #         aspects[i//2]['to'] = new_aspect_token_range[i+1]
                #         if words[aspects[i//2]['from']:aspects[i//2]['to']] != aspects[i//2]['term']:
                #             print(f"{words[aspects[i//2]['from']:aspects[i//2]['to']]} -> {aspects[i//2]['term']}")
                 
                samples.append({
                    'idx': len(samples),
                    'aspects': aspects,
                    'tokens': tokens,
                    'sentence_raw': sentence,
                    'sentence': ' '.join(tokens),
                    # 'tokens': words,
                    # 'heads': heads,
                    # 'deprels': deprels,
                    # 'tags': tags,
                })

            samples_json = json.dumps(samples, indent=4)
            with open(f'{path}{desc}.multiple.json', 'w') as fw:
                fw.write(samples_json)
            
            return samples
    
    def get_vector(self, args=None, tokenizer=None, only=None):      
        def get_embedding(sample, tokenizer):
            tokens, ids, mask_asp = sample['tokens'], [], []
            for i, token in enumerate(tokens):
                idx = tokenizer.encode(token, return_tensors='pt', add_special_tokens=False)
                ids.extend(idx)
                if i>=sample['aspect']['from'] and i<sample['aspect']['to']:
                    mask_asp.extend([1]*len(idx))
                else:
                    mask_asp.extend([0]*len(idx))
            
            assert len(ids) == len(mask_asp)
            return {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([1]*len(ids)),
                'attention_mask_asp': torch.tensor(mask_asp),
            }

        self.tokenizer = tokenizer
        for desc, samples in self.datas['text'].items():
            if only is not None and desc!=only: continue
            samples_embed = []
            for sample in samples:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                embedding = get_embedding(sample, tokenizer)
                sample_embed = {
                    'idx': len(samples_embed),
                    'input_ids': embedding['input_ids'],
                    'attention_mask': embedding['attention_mask'],
                    'attention_mask_asp': embedding['attention_mask_asp'],
                    'polarity': self.tokenizer_['pols'].vocab[sample['aspect']['polarity']],
                }
                samples_embed.append(sample_embed)

            self.datas['vector'][desc] = samples_embed


        self.args, self.tokenizer = args, tokenizer
        self.mask_token, self.mask_token_id = tokenizer.mask_token, tokenizer.mask_token_id
        self.eos_token, self.eos_token_id = tokenizer.eos_token, tokenizer.eos_token_id
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                query, labels = '', []
                for aspect in item['aspects']:
                    if query != '': query += f' {self.eos_token} '
                    query += f"the sentiment of {aspect['term']} is {self.mask_token}"
                    labels.append(aspect['polarity'])

                embedding = tokenizer.encode_plus(item['text'], query, return_tensors='pt')
                item_embed = {
                    'idx': item['idx'],
                    'input_ids': embedding.input_ids.squeeze(dim=0),
                    'attention_mask': embedding.attention_mask.squeeze(dim=0),
                    'token_type_ids': embedding.token_type_ids.squeeze(dim=0),
                    'polarity_ids': torch.tensor(labels),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def split_dev(self, rate=0.2, method='random'):
        samps_len = len(self.datas['text']['train'])
        valid_len = int(samps_len*rate); train_len = samps_len-valid_len

        if method == 'random':
            sel_index = list(range(samps_len)); random.shuffle(sel_index)
            train = [self.datas['text']['train'][index] for index in sel_index[0:train_len]]
            valid = [self.datas['text']['train'][index] for index in sel_index[-valid_len:]]

        for i, sample in enumerate(train): sample['idx'] = i
        for i, sample in enumerate(valid): sample['idx'] = i

        self.datas['text']['train'] = train
        self.datas['text']['valid'] = valid

    def get_dataloader(self, batch_size, shuffle=None, collate_fn=None, only=None, split=None):
        if shuffle is None: shuffle = {'train': True, 'valid': False, 'test': False}
        if collate_fn is None: collate_fn = self.collate_fn

        dataloader = {}
        for desc, data_embed in self.datas['data'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=collate_fn)
            
        if only is None and 'valid' not in dataloader: dataloader['valid'] = dataloader['test']
        return dataloader

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

class ABSADataset(ABSADataset_MA):
    def __init__(self, path, tokenizer=None, lower=False):
        self.lower = lower
        self.name = ['absa', path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        if not os.path.exists(path+'train.multiple.json'): ABSADataset_MA(path, lower=lower)
        for desc in ['train', 'test']:
            self.datas['data'][desc] = self.get_dataset(path, desc) # 解析数据集
        self.datas['data']['valid'] = self.datas['data']['test']
        self.get_tokenizer_(self.datas['data']['train'], names=['polarity']) # 获取 tokenizer
        self.n_class = len(self.tokenizer_['labels']['ltoi'])

    def get_dataset(self, path, desc):
        def get_distance(sample):
            dis = {'left':[], 'mid': [], 'right':[]}
            for i, token in enumerate(sample['tokens']):
                if i < sample['aspect']['from']: dis['left'].extend([i-sample['aspect']['from']])
                if i >= sample['aspect']['to']: dis['right'].extend([i+1-sample['aspect']['to']])
                if i>=sample['aspect']['from'] and i<sample['aspect']['to']: dis['mid'].extend([0])
            return dis['left']+dis['mid']+dis['right']

        raw_path = f'{path}/{desc}.multiple.json'
        if not os.path.exists(raw_path): return None
        with open(raw_path, 'r', encoding='utf-8') as fp:
            raw_samples, samples = json.load(fp), []
        for sample in tqdm(raw_samples):
            aspects = sample['aspects']
            for aspect in aspects:
                temp = copy.deepcopy(sample)
                temp['idx'] = len(samples)
                temp['aspect'] = ' '.join(aspect['term'])
                temp['aspect_pos'] = [aspect['from'], aspect['to']]
                if ' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]]) != temp['aspect']:
                    print(f"{' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]])} -> {temp['aspect']}")
                temp['polarity'] = aspect['polarity']
                #temp['distance'] = get_distance(temp)
                del temp['aspects']
                samples.append(temp)

        return samples

    def get_vector(self, tokenizer, only=None, is_dep=False): 
        def get_adjs(seq_len, sample_embed, self_token_id, directed=True, loop=False):
            # head -> adj (label: dep)
            idx_asp = [idx for idx, val in enumerate(sample_embed['attention_mask_asp']) if val]
            heads, deps = sample_embed['dep_heads'], sample_embed['dep_deprels']
            adjs = np.zeros((seq_len, seq_len), dtype=np.float32)
            edges = np.zeros((seq_len, seq_len), dtype=np.int64)
            # head 指向 idx：第i个token的父节点是第i个head对应位置的节点
            for idx, head in enumerate(heads):
                if idx in idx_asp: # 是 aspect
                    for k in idx_asp: 
                        adjs[idx, k], edges[idx, k] = 1, self_token_id
                        adjs[k, idx], edges[k, idx] = 1, self_token_id
                if head != -1: # non root
                    adjs[head, idx], edges[head, idx] = 1, deps[idx]
                if not directed: # 无向图
                    adjs[idx, head], edges[idx, head] = 1, deps[idx]
                if loop: # 自身与自身相连idx
                    adjs[idx, idx], edges[idx, idx] = 1, self_token_id
            
            sample_embed['dep_graph_adjs'] = torch.tensor(adjs)
            sample_embed['dep_graph_edges'] = torch.tensor(edges)
            return sample_embed

        def get_dependency(sample, sample_embed, tokenizers):
            sample_embed['dep_heads'] = torch.tensor(sample['heads'])
            for desc in ['deprels', 'tags']:
                temp, vocab, unk_token = sample[desc], tokenizers[desc].vocab, tokenizers[desc].unk_token
                temp_id = [vocab[unk_token] if item not in vocab else vocab[item] for item in temp]
                sample_embed['dep_'+desc] = torch.tensor(temp_id)
            
            dep_self_token_id = tokenizers['deprels'].vocab[tokenizers['deprels'].self_token]
            sample_embed = get_adjs(len(sample_embed['dep_heads']), sample_embed, dep_self_token_id)
            return sample_embed
        
        def get_embedding(sample, tokenizer):
            tokens, ids, mask_asp = sample['tokens'], [], []
            dis_asp = {'left':[], 'mid': [], 'right':[]}
            for i, token in enumerate(tokens):
                idx = tokenizer.encode(token, return_tensors='pt', add_special_tokens=False)
                ids.extend(idx)
                # context-aspect distance
                if i < sample['aspect_pos'][0]: dis_asp['left'].extend([i-sample['aspect_pos'][0]]*len(idx))
                if i >= sample['aspect_pos'][1]: dis_asp['right'].extend([i+1-sample['aspect_pos'][1]]*len(idx))
                # aspect mask
                if i>=sample['aspect_pos'][0] and i<sample['aspect_pos'][1]:
                    dis_asp['mid'].extend([0]*len(idx))
                    mask_asp.extend([1]*len(idx))
                else:
                    mask_asp.extend([0]*len(idx))

            assert len(ids) == len(mask_asp)
            distance_asp = dis_asp['left']+dis_asp['mid']+dis_asp['right']
            return {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([1]*len(ids)),
                'attention_mask_asp': torch.tensor(mask_asp),
                'asp_dis_ids': torch.tensor([dis+100 for dis in distance_asp])
            }
                
        self.tokenizer = tokenizer
        for desc, samples in self.datas['text'].items():
            if samples is None: continue
            if only is not None and desc!=only: continue
            samples_embed = []
            for sample in samples:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                embedding = get_embedding(sample, tokenizer)
                sample_embed = {
                    'idx': sample['idx'],
                    'input_ids': embedding['input_ids'],
                    'attention_mask': embedding['attention_mask'],
                    'attention_mask_asp': embedding['attention_mask_asp'],
                    'asp_dis_ids': embedding['asp_dis_ids'],
                    'label': self.tokenizer_['polarity'].vocab[sample['polarity']],
                }
                if is_dep: sample_embed = get_dependency(sample, sample_embed, tokenizers=self.tokenizer_) # 解析句法依赖信息
                sample['label'] = sample_embed['label']
                samples_embed.append(sample_embed)

            self.datas['vector'][desc] = samples_embed

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def get_specific_dataset(args, d_type='single'):
    ## 1. 导入数据
    data_path = args.file['data_dir'] + f"{args.train['tasks'][1]}/"
    if d_type == 'multi':
        dataset = ABSADataset_MA(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }
    else:
        dataset = ABSADataset(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }

    dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    for desc, data in dataset.datas['data'].items():
        dataset.datas['data'][desc] = DataLoader_ABSA(
            dataset,
            d_type=d_type,
            desc=desc
        )
    dataset.task = 'cls'

    return dataset


def parse_xml(path):
    sentences, samples = ET.parse(path).getroot(), []
    for sentence in sentences:
        for item in sentence:
            if item.tag == 'text':
                sample = {'text': item.text, 'aspects': []}
            else:
                for asp in item:
                    term, polarity = asp.attrib['term'], asp.attrib['polarity']
                    position = [int(asp.attrib['from']), int(asp.attrib['to'])]
                    assert sample['text'][position[0]:position[1]] == term
                    sample['aspects'].append({ 'term': term, 'polarity': polarity, 'position': position })
                
                asp_begins = [aspect['position'][0] for aspect in sample['aspects']]
                if not all(x<=y for x, y in zip(asp_begins, asp_begins[1:])):
                    modify_index = sorted(range(len(asp_begins)), key=lambda x:asp_begins[x])
                    sample['aspects'] = [sample['aspects'][idx] for idx in modify_index]

                samples.append(sample) # 有aspect才存储

    return samples



class ABSADataModule(Dataset):
    def __init__(self, data_dir, batch_size=2, num_workers=8) -> None:
        super().__init__()
        self.name = ['absa', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # dataset initialize
        self.prepare_data(stages=['train', 'test']) # 
        self.get_tokenizer_(self.datas['train'], names=['polarity'])
        self.num_classes = len(self.tokenizer_['labels']['l2i'])

        self.datas['valid'] = self.datas['test'] # no validation set

    def dataset_init(self):
        self.info = {
            'max_seq_token_num': {}, # 句子 最长长度
            'max_asp_token_num': {}, # aspect 最长长度
            'total_samples_num': {}, # 样本数量
            'class_category': {},    # 类别统计
        }

        # 初始化数据集要保存的内容 
        self.datas = {}
        self.loader = {}

        # tokenizer
        self.tokenizer_ = {
            'labels': { 'l2i': {}, 'i2l': {}, 'count': {} }
        }

    def get_tokenizer_(self, samples, names):
        if not isinstance(names, list): names = [names]
        if 'polarity' in names:
            for samp in samples:
                value = samp['polarity']
                if value not in self.tokenizer_['labels']['l2i']:
                    self.tokenizer_['labels']['l2i'][value] = len(self.tokenizer_['labels']['l2i'])
                    self.tokenizer_['labels']['i2l'][len(self.tokenizer_['labels']['i2l'])] = value
                    self.tokenizer_['labels']['count'][value] = 1
                self.tokenizer_['labels']['count'][value] += 1

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}.multiple.json'
            if not os.path.exists(raw_path): return None
            with open(raw_path, 'r', encoding='utf-8') as fp: raw_samples, samples = json.load(fp), []

            for sample in tqdm(raw_samples):
                aspects = sample['aspects']
                for aspect in aspects:
                    temp = copy.deepcopy(sample)
                    temp['index'] = len(samples)
                    temp['aspect'] = ' '.join(aspect['term'])
                    temp['aspect_pos'] = [aspect['from'], aspect['to']]
                    if ' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]]) != temp['aspect']:
                        print(f"{' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]])} -> {temp['aspect']}")
                    temp['polarity'] = aspect['polarity']
                    samples.append(temp)

            self.datas[stage] = samples

    def setup(self, tokenizer, stage=None):
        self.tokenizer = tokenizer
        for stage, samples in self.datas.items():
            if samples is None: continue
            self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                # asp_pos = sample['aspect_pos']
                # sentence = ' '.join(sample['tokens'][0:asp_pos[0]]) + ' [' + sample['aspect'] + '] ' + ' '.join(sample['tokens'][asp_pos[1]:])
                # aspect = '[' + sample['aspect'] + ']'
                # embedding = tokenizer.encode_plus(sentence, aspect, return_tensors='pt')

                embedding = tokenizer.encode_plus(sample['sentence'], sample['aspect'], return_tensors='pt')
                sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(dim=0)
                sample['token_type_ids'] = embedding['token_type_ids'].squeeze(dim=0)
                sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                
                self.info['class_category'][stage][sample['label']] += 1

    def get_dataloader(self, batch_size=None):
        if batch_size: self.batch_size = batch_size
        for stage, _ in self.datas.items():
            if stage=='train': self.loader[stage] = self.train_dataloader()
            if stage=='valid': self.loader[stage] = self.val_dataloader()
            if stage=='test':  self.loader[stage] = self.test_dataloader()
        return self.loader

    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, samples):
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs
    

if __name__ == '__main__':
    data_dir = '/home/jzq/My_Codes/CodeFrame/Datasets/Textual/absa/twi/'
    dataset = ABSADataset_MA(data_dir, lower=True)


    dataset = ABSADataModule(data_dir)
    plm_dir = None
    tokenizer = AutoTokenizer.from_pretrained(plm_dir)
    dataset.setup(tokenizer)