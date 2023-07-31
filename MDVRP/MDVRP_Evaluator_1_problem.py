import torch
from logging import getLogger
from MDVRP_EnvEval import MDVRPEnvEval
from MDVRP_Model import MDVRPModel
from my_utils import *
from MDVRP_Draw_1_problem import Draw_1_Problem
import copy
import numpy as np

class MDVRPEvaluator:
    def __init__(self,
                 env_params,
                 model_params,
                 eval_params):

        # params
        self.env_params = env_params
        self.model_params = model_params
        self.eval_params = eval_params
        self.batch_size = self.eval_params['batch_size']
        self.augmentation = self.eval_params['augmentation']
        self.sample = self.eval_params['sample']
        self.sample_size = self.batch_size if self.sample else 1
        self.data = None
        # result, log
        self.logger = getLogger(name='evaluator')
        self.result_folder = get_result_folder()

        # cuda
        use_cuda = self.eval_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.eval_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # main components
        self.env = MDVRPEnvEval(**self.env_params)
        self.model = MDVRPModel(**self.model_params)

        # restore
        self.model_load = self.eval_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**self.model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score = AverageMeter()
        score_aug = AverageMeter()

        score, score_aug = self.eval_1_sample_batch()

        # logs
        self.logger.info('Evaluate done, score without aug: {:.4f}, score with aug: {:.4f}'
                         .format(score, score_aug))

    def eval_1_sample_batch(self):
        # aug
        if self.augmentation['aug_d']:
            aug_type = 'd'
        elif self.augmentation['aug_8']:
            aug_type = '8'
        else:
            aug_type = None

        # prepare
        self.model.eval()
        with torch.no_grad():
            self.data = self.env.load_dataset_problems(self.sample_size, self.sample, aug_type)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        aug_factor = self.data['aug_factor']

        # mt rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch_size, mt_size)
            state, reward, done = self.env.step(selected)

        # score
        reward = reward.reshape(aug_factor, -1, self.env.mt_size)
        # shape: (aug_factor, sample, mt_size)
        solution = state.result_node_list.reshape(aug_factor, self.sample_size * self.env.mt_size, -1)
        # shape: (aug_factor, sample, mt_size, solution)
        reward_max_mt, max_index_1 = reward.reshape(aug_factor, -1).max(dim=1)
        print(solution.size())
        # shape: (aug_factor)
        print(max_index_1)
        solution_max_mt = (torch.gather(solution, dim=1, index=max_index_1[:, None, None].expand(
                aug_factor, -1, solution.size(2)))).squeeze(1)
        # shape: (aug_factor, solution)
        reward_max_aug_mt, max_index_2 = reward_max_mt.max(dim=0)
        # shape : (1)
        solution_max_aug_mt = torch.index_select(solution_max_mt, dim=0, index=max_index_2)
        print(solution_max_aug_mt.size())
        # shape: (solution)
        self.logger.info('route:')
        self.logger.info('[{}]'.format(solution_max_aug_mt.cpu().numpy().tolist()))
        draw_data = copy.deepcopy(self.data)
        draw_data['depot_x_y'] = copy.deepcopy(self.data['depot_x_y'][max_index_2])
        draw_data['customer_x_y'] = copy.deepcopy(self.data['customer_x_y'][max_index_2])
        draw_data['full_node'] = np.vstack((draw_data['depot_x_y'], draw_data['customer_x_y']))
        is_depot = np.zeros([len(draw_data['depot_x_y'] + draw_data['customer_x_y']), 2])
        for i in range(len(draw_data['depot_x_y'])):
            is_depot[i] = [1, 1]
        draw_data['full_node'] = np.hstack((draw_data['full_node'], is_depot))

        Draw_1_Problem(draw_data, solution_max_aug_mt, self.result_folder)
        score_no_aug = -reward_max_mt[0].float() * self.data['scale']
        score_aug = -reward_max_aug_mt.float() * self.data['scale']
        return score_no_aug.item(), score_aug.item()
