import torch, time, logging
from tqdm import tqdm

from utils_model import init_weight, print_trainable_parameters
from utils_processor import Metrics, get_optimizer, get_scheduler


class Processor():
    def __init__(self, args, model, dataset) -> None:
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.train['device'])
        self.model.metrics = Metrics(args, dataset)
        init_weight(self.model) # 初始化模型参数
        print_trainable_parameters(self.model) # 打印训练参数比重

        self.log_step_rate = args.train['log_step_rate']
        self.global_step = 1

        for k, v in vars(args).items():
            for kk, vv in v.items(): args.logger['params'].info(f"{k}.{kk}: {vv}")
        args.logger['params'].info("\n ====================================== \n")

        display = ''
        for item in args.train['display']: 
            if item in args.train: display += f"{item}: {args.train[item]}, "
                # args.logger['process'].warning(f"{item}: {args.train[item]}")
            if item in args.model: display += f"{item}: {args.model[item]}, "
                # args.logger['process'].warning(f"{item}: {args.model[item]}")
        args.logger['process'].warning(display)

    def loadState(self, iter=0, bl_rate=0.9):
        if self.dataset.loader: self.dataloader = self.dataset.loader
        else: self.dataloader = self.dataset.get_dataloader(self.args.train['batch_size'])
        iter_total = int(len(self.dataloader['train'])*self.args.train['epochs']) - iter
        self.optimizer = get_optimizer(self.model)
        self.scheduler = get_scheduler(self.args, self.optimizer, iter_total)

        # load checkpoint
        # if self.args.train['inference']: # 
        #     checkpoint = torch.load(self.save_path+f'best.state')
        #     self.model.load_state_dict(checkpoint['net'])
            # self.best_result = checkpoint['result']
            # self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_desc(self, epoch, ttime=None):
        args, metrics = self.args, self.model.metrics
        epochs, model_name, data_name = args.train['epochs'], args.model['name'], self.dataset.name[-1]
        m = self.dataset.metric
        m_tr, m_vl, m_te = round(metrics.train[m], 3), round(metrics.valid[m], 3), round(metrics.test[m], 3)
        m_tr_loss = round(metrics.train['loss'], 3)
        desc = f"eh {epoch}/{epochs} ({model_name}=>{data_name}: {str(m_tr)}/{str(m_vl)}/{str(m_te)}, loss: {str(m_tr_loss)}, time: {ttime})"
        self.tqdm_epochs.set_description(desc)
        if epoch>=0: self.tqdm_epochs.update()

    def train_stop(self, epoch=None):
        args = self.args

        # 1. 长期未更新了，增加评价次数
        early_threshold = epoch - self.model.metrics.valid['epoch']
        if early_threshold >= args.train['early_stop']:
            return True

        # self.log_step_rate = (self.params.log_step_rate+early_threshold)/1.3 # for absa
        if early_threshold: 
            self.log_step_rate = args.train['log_step_rate']+early_threshold*0.5
            self.log_step_rate = min(self.log_step_rate, 3.0)
        else: self.log_step_rate = args.train['log_step_rate']

    def train_epoch(self):
        log_step = int(len(self.dataloader['train']) / self.log_step_rate)
        torch.cuda.empty_cache()
        #for batch in tqdm(self.dataloader['train'], smoothing=0.05):
        for bi, batch in enumerate(self.dataloader['train']):
            self.model.train()      
            for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
            outs = self.model.training_step(batch, bi)  
            
            outs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train['max_grad_norm'])
            self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step() 
            self.optimizer.zero_grad()        

            self.global_step += 1
            if self.global_step % log_step == 0: 
                self._evaluate(stage='valid')
                if self.args.train['do_test'] and self.model.valid_update: self._evaluate(stage='test')
                if self.args.train['save_model']:  
                    state = {
                        'net': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'step': self.global_step,
                        # 'result': self.model.metrics,
                    }
                    torch.save(state, self.save_path+f'best.state_')

    def _train(self):
        self.loadState()
        epochs = self.args.train['epochs']
        self.tqdm_epochs = tqdm(total=epochs, position=0) # 进度条
        self.train_desc(epoch=-1) # initialize process bar
        if self.args.model['epoch_before']: self.model.epoch_deal()
        for epoch in range(0, epochs):
            s_time = time.time()
            self.model.cur_epoch = epoch
            if self.args.model['epoch_every']: self.model.epoch_deal()
            self.train_epoch()
            if self.args.model['epoch_after']: self.model.epoch_deal()
            self.model.on_train_epoch_end()

            self.train_desc(epoch, round(time.time()-s_time, 1))
            if self.train_stop(epoch): break 
            
        self.tqdm_epochs.close()
        return self.model.metrics

    def _evaluate(self, stage='test'):
        for bi, batch in enumerate(self.dataloader[stage]):
            self.model.eval()
            with torch.no_grad():
                for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
                if stage == 'valid': self.model.validation_step(batch, bi)
                if stage == 'test': self.model.test_step(batch, bi)
            
        if stage == 'valid': self.model.on_validation_end()
        if stage == 'test': self.model.on_test_end()
        return self.model.metrics