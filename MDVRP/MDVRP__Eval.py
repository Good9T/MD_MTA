##########################################################################################
# import

import os
import sys
import logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for vrp_utils
sys.path.append('home/ljq21/md-mta/my_utils')
from my_utils import create_logger, copy_all_src
from MDVRP_Evaluator import MDVRPEvaluator as evaluator1
from MDVRP_Evaluator_1_problem import MDVRPEvaluator as evaluator2

##########################################################################################
# Machine Environment Config
debug_mode = False
use_cuda = True
cuda_device_num = 0
copy_source = False
##########################################################################################
# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
##########################################################################################
# parameters
logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}

env_params = {
    'customer_size': 100,
    'mt_size': 102,
    'depot_size': 3,
    'capacity': 50,
    'demand_min': 1,
    'demand_max': 10,
    'load_path': '',
    'synthetic_dataset': True,
    'sample': True,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'clip': 10,
    'ff_hidden_dim': 512,
    'sample':  True if env_params['sample'] else False,  # argmax, softmax
}

eval_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'model_load': {
        'path': './result/',  # directory path of model and log files saved.
        'epoch': 1000,  # epoch version of pre-trained model to load.
    },
    'episodes': 2000,
    'sample':  True if env_params['sample'] else False,
    'batch_size': 160,
    'augmentation': {
        'aug_d': True,
        'aug_8': False,
    }
}


##########################################################################################
# main
def main():
    if debug_mode:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    if env_params['synthetic_dataset']:
        evaluator = evaluator1(env_params=env_params,
                               model_params=model_params,
                               eval_params=eval_params)
    else:
        evaluator = evaluator2(env_params=env_params,
                               model_params=model_params,
                               eval_params=eval_params)

    if copy_source:
        copy_all_src(evaluator.result_folder)

    evaluator.run()


def _set_debug_mode():
    global eval_params
    eval_params['episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(debug_mode))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(use_cuda, cuda_device_num))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
