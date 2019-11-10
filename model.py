## has avaliable
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
from torch.autograd import Variable
from Config import Config
import math

ctx_dim = (math.ceil(Config.IMAGE_WIDTH/4)**2, 512)



class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 512, 3, stride=1, padding=1)

        self.Wai = nn.Linear(ctx_dim[1], ctx_dim[1], bias=False)
        self.Wh = nn.Linear(512, ctx_dim[1], bias=False)
        self.att = nn.Linear(ctx_dim[1], 1)

        # self.fc = nn.Linear(ctx_dim[1] * 4 * 4, 256)

        self.lstm = nn.LSTMCell(ctx_dim[1], 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)

        self.Wai.weight.data = norm_col_init(self.Wai.weight.data, 1.0)

        self.Wh.weight.data = norm_col_init(self.Wh.weight.data, 1.0)

        self.att.weight.data = norm_col_init(self.att.weight.data, 1.0)
        self.att.bias.data.fill_(0)

        # self.fc.weight.data = norm_col_init(self.fc.weight.data, 1.0)
        # self.fc.bias.data.fill_(0)

        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))                             
        x = F.relu(self.conv4(x))                             # [B, ctx_dim[1], 4 , 4]
        filter_size = (x.size()[2])
        ai = x.view(-1, ctx_dim[1], ctx_dim[0])               # [B, ctx_dim[1], 16]
        ai = ai.transpose_(1, 2)                              # [B, 16, ctx_dim[1]]
        ai_ = ai.view(-1, ctx_dim[1])                         # [B*16, ctx_dim[1]]
        Lai = self.Wai(ai_)
        Lai = Lai.view(-1, ctx_dim[0], ctx_dim[1])            # [B, 16, ctx_dim[1]]

        Uh_ = self.Wh(hx)
        Uh_ = torch.unsqueeze(Uh_, 1)                         # [B, 1, ctx_dim[1]]

        Lai_Uh_ = (torch.add(Lai, Uh_)).view(-1, ctx_dim[1])  # [B*16, ctx_dim[1]]
        att_ = self.att(torch.tanh(Lai_Uh_))                  # [B*16, 1]

        alpha_ = F.softmax(att_.view(-1, ctx_dim[0]), dim=1)          # [B, 16]
        zt = torch.sum(torch.mul(ai, torch.unsqueeze(alpha_, 2)), 1)  # [B, ctx_dim[1]]

        alpha_reshape = alpha_.view(filter_size, filter_size)
        #context = self.fc(zt)
        # x = x.view(-1, 32 * 4 * 4)
        # x = x.view(x.size(0), -1)  # [1,512]
        # x = self.fc(x)

        hx, cx = self.lstm(zt, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), alpha_reshape
