import torch
from logging import getLogger
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from my_utils import *
from MDOVRP_Model import MDOVRPModel
from MDOVRP_EnvTrain import MDOVRPEnvTrain


class MDOVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.epochs = self.trainer_params['epochs']
        self.episodes = self.trainer_params['episodes']
        self.model_save_interval = self.trainer_params['logging']['model_save_interval']
        self.img_save_interval = self.trainer_params['logging']['img_save_interval']

        # result, log
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        use_cuda = self.trainer_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # main components
        self.model = MDOVRPModel(**model_params)
        self.env = MDOVRPEnvTrain(**env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # restore
        self.start_epoch = 1
        self.model_load = trainer_params['model_load']
        if self.model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**self.model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + self.model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = self.model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded!!')

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.epochs + 1):

            # Train
            score, loss = self.train_1_epoch(epoch)
            self.result_log.append('score', epoch, score)
            self.result_log.append('loss', epoch, loss)

            # LR decay
            self.scheduler.step()

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.epochs)
            self.logger.info('Epoch {:3d}/{:3d}: Time Estimate: Elapsed: [{}] Remain[{}]'
                             .format(epoch, self.epochs, elapsed_time_str, remain_time_str))

            all_done = (epoch == self.epochs)

            # Save images of every latest epoch
            if epoch > 1:
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['loss'])

            # Save model
            if all_done or (epoch % self.model_save_interval) == 0:
                self.logger.info('Saving model')
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save image
            if all_done or (epoch % self.img_save_interval) == 0:
                self.logger.info('Saving image')
                image_prefix = '{}/image/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['loss'])

            # All done
            if all_done:
                self.logger.info('Training done, printing log array')
                util_print_log_array(self.logger, self.result_log)

    def train_1_epoch(self, epoch):
        score = AverageMeter()
        loss = AverageMeter()

        episode = 0
        while episode < self.episodes:
            remain_episodes = self.episodes - episode
            batch_size = min(self.trainer_params['batch_size'], remain_episodes)
            score_avg, loss_avg = self.train_1_batch(batch_size)
            score.update(score_avg, batch_size)
            loss.update(loss_avg, batch_size)
            episode += batch_size

        # log
        self.logger.info('Epoch: {:3d}  Score: {:.4f}, Loss :{:.4f}'
                         .format(epoch, score.avg, loss.avg))
        return score.avg, loss.avg

    def train_1_batch(self, batch_size):
        # prepare
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        prob = torch.zeros(size=(batch_size, self.env.mt_size, 0))
        # shape: (batch_size, mt_size, 0~)

        # multi_trajectories rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, probability = self.model(state)
            # shape: (batch_size, mt_size)
            state, reward, done = self.env.step(selected)
            prob = torch.cat((prob, probability[:, :, None]), dim=2)

        # loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch_size, mt_size)
        log_prob = prob.log().sum(dim=2)
        # shape: (batch_size, mt_size)
        loss = -advantage * log_prob
        # shape: (batch_size, mt_size)
        loss_mean = loss.mean()

        # score
        max_mt_reward, _ = reward.max(dim=1)
        score_mean = -max_mt_reward.float().mean()

        # step
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()
