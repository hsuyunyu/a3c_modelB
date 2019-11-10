from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Config import Config


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.available = None
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.alpha = []
        self.x = []
        self.envs_mean = None
        self.envs_std = None
        self.log_epsilon = Config.LOG_EPSILON
        #self.action_all = []


    def action_train(self):
        # if self.gpu_id >= 0:
        #     with torch.cuda.device(self.gpu_id):
        #         self.log_epsilon = torch.tensor(Config.LOG_EPSILON).cuda()
        #         self.action_index = Variable(torch.zeros(1, 16).cuda())
        # else:
        #     self.log_epsilon = torch.tensor(Config.LOG_EPSILON)
        #     self.action_index = Variable(torch.zeros(1, 16))


        value, logit, (self.hx, self.cx), _ = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        # prob = F.softmax((((logit - self.available) + Config.MIN_POLICY)/(1.0 + Config.MIN_POLICY * 16)), dim=1)
        # prob = F.softmax(logit - self.available, dim=1)
        prob = F.softmax(logit, dim=1)


        log_prob = F.log_softmax(prob, dim=1)

        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data  # choose action
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, available,  self.envs_mean, self.envs_std = self.env.step(
            action.cpu().numpy())

        self.info = self.envs_mean  # ##

        self.state = torch.from_numpy(state).float()
        self.available = torch.from_numpy(available).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
                self.available = self.available.cuda()
        self.reward = max(min(self.reward, 1), -1)  # limit reward
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx), alpha_reshape = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        if Config.GIF:
            self.alpha.append(alpha_reshape.data.cpu().numpy())
            self.x.append(self.state.view(22, 22).data.cpu().numpy())

        prob = F.softmax(logit, dim=1)
        # prob = F.softmax(logit-self.available, dim=1)
        # prob = F.softmax((((logit - self.available) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * 16)), dim=1)

        #  choose random action
        action = prob.multinomial(1).data
        state, self.reward, self.done, available,  self.envs_mean, self.envs_std = self.env.step(
            action.cpu().numpy())

        #  choose max prob action
        # action = prob.max(1)[1].data.cpu().numpy()
        # state, self.reward, self.done, available,  self.envs_mean, self.envs_std = self.env.step(action[0])

        self.info = self.envs_mean  # print information

        self.state = torch.from_numpy(state).float()
        self.available = torch.from_numpy(available).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
                self.available = self.available.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        return self
