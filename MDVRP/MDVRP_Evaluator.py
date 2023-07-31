import torch
from logging import getLogger
from MDVRP_EnvEval import MDVRPEnvEval
from MDVRP_Model import MDVRPModel
from my_utils import *


class MDVRPEvaluator:
    def __init__(self,
                 env_params,
                 model_params,
                 eval_params):

        # params
        self.env_params = env_params
        self.model_params = model_params
        self.eval_params = eval_params
        self.episodes = self.eval_params['episodes']
        self.batch_size = self.eval_params['batch_size']
        self.augmentation = self.eval_params['augmentation']
        self.sample = self.eval_params['sample']
        self.sample_size = self.batch_size

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
        episode = 0

        while episode < self.episodes:
            remain_episode = self.episodes - episode
            if self.sample:
                batch = 1
                score_avg, score_aug_avg = self.eval_1_sample()
            else:
                batch = min(self.batch_size, remain_episode)
                score_avg, score_aug_avg = self.eval_1_batch(batch)
            score.update(score_avg, batch)
            score_aug.update(score_aug_avg, batch)
            episode += batch

            # logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, self.episodes)
            self.logger.info('episode: {:3d}/{:3d}, Elapsed: [{}], Remain: [{}], score: {:.3f}, score_aug: {:.3f}'
                             .format(episode, self.episodes, elapsed_time_str, remain_time_str, score_avg,
                                     score_aug_avg))
            all_done = (episode == self.episodes)
            if all_done:
                self.logger.info('Evaluate done, score without aug: {:.4f}, score with aug: {:.4f}'
                                 .format(score.avg, score_aug.avg))

    def eval_1_batch(self, batch):
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
            self.env.load_random_problems(batch, self.sample, aug_type)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # mt rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch_size, mt_size)
            state, reward, done = self.env.step(selected)

        # score
        reward_aug = reward.reshape(-1, batch, self.env.mt_size)

        # shape: (aug_factor, batch_size, mt_size)
        reward_max_mt, _ = reward_aug.max(dim=2)
        # shape: (aug_factor, batch_size)
        reward_max_aug_mt, _ = reward_max_mt.max(dim=0)
        # shape : (batch_size)

        score_no_aug = -reward_max_mt[0, :].float().mean()
        score_aug = -reward_max_aug_mt.float().mean()

        return score_no_aug.item(), score_aug.item()

    def eval_1_sample(self):
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
            self.env.load_random_problems(self.sample_size, self.sample, aug_type)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # mt rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch_size, mt_size)
            state, reward, done = self.env.step(selected)

        # score
        reward_aug = reward.reshape(-1, self.env.mt_size)
        # shape: (aug_factor * sample_size, mt_size)
        reward_max_mt, _ = reward_aug.max(dim=1)
        # shape: (aug_factor * sample_size)
        reward_max_aug_mt, _ = reward_max_mt.max(dim=0)
        # shape : (1)
        score_no_aug = -reward_max_mt[0].float()

        score_aug = -reward_max_aug_mt

        return score_no_aug.item(), score_aug.item()
