
## cuda environment
import warnings, logging, os, wandb, torch
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='false'


from config import config
from processor import Processor
from utils_processor import set_rng_seed

def run(args):
    if args.train['wandb']:
        wandb.init(
            project=f"project: {'-'.join(args.train['tasks'])}",
            name=f"{'-'.join(args.train['tasks'])}-seed-{args.train['seed']}",
        )
    set_rng_seed(args.train['seed']) #

    # import model and dataset
    from Model_AGL_ import import_model
    model, dataset = import_model(args)

    # train or eval the model
    processor = Processor(args, model, dataset)
    if args.train['inference']:
        processor.loadState()
        model.load_state_dict(
            torch.load("checkpoints/rest/save_params_1.pth"), 
            strict=False)
        model.asp_embedding = torch.load("checkpoints/rest/save_embedding_1.pth")
        result = processor._evaluate(stage='test')
        print(result.test)
    else: result = processor._train()
    if args.train['wandb']: wandb.finish()

    ## 2. output results
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr':      args.train['learning_rate'],
            'lr_pre':  args.train['learning_rate_pre'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
            'ccl':     args.model['loss_ccl'],
        },
        'metric': {
            'stop':    result.valid['epoch'],
            'tr_mf1':  result.train[dataset.metric],
            'tv_mf1':  result.valid[dataset.metric],
            'te_mf1':  result.test[dataset.metric],
        },
    }
    return record


if __name__ == '__main__':
    args = config(task='absa', dataset='rest', framework=None, model='agl')

    args.model['scale'] = 'large'
    args.train['device_ids'] = [0]
    
    args.train['epochs'] = 25
    args.train['early_stop'] = 10
    args.train['batch_size'] = 32
    args.train['log_step_rate'] = 1.0
    args.train['learning_rate'] = 3e-4
    args.train['learning_rate_pre'] = 3e-4
    args.train['save_model'] = 0
    args.train['inference'] = 1 
    args.train['do_test'] = False
    args.train['wandb'] = 0 # True   
    
    args.train['log_step_rate_max'] = 5
    args.model['drop_rate'] = 0.3
    args.model['use_adapter'] = 1
    args.model['loss_agl'] = 1.0
    args.model['loss_ccl'] = 1.0
    args.train['display'].extend(['loss_agl', 'loss_ccl'])

    seeds = [1,2,3]
    if seeds or args.train['inference']:
        if not seeds: seeds = [args.train['seed']]
        for seed in seeds:
            args.train['seed'] = seed
            record = run(args)
