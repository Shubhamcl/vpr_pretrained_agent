from asyncio import futures
import torch
from torch.nn import functional as F

class BranchedLoss():
    def __init__(self):
        self.loss = F.l1_loss
        self.branch_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.action_weights = [0.5, 0.5]
        self.speed_weight = 0.05

        self.n_branches = len(self.branch_weights)
        self.n_actions = len(self.action_weights)
        
    def forward(self, outputs, supervisions, commands):
        # Clamp to index branches from 0
        commands.clamp_(0, self.n_branches-1)

        # [[ Action loss ]]
        branch_masks = self._get_branch_masks(commands, n_branches=self.n_branches, \
            n_actions=self.n_actions)
        action_loss = 0.0
        for i in range(self.n_branches):
            masked_action_pred = outputs['action_branches'][i]*branch_masks[i]
            masked_action = supervisions['action']*branch_masks[i]
            for j in range(self.n_actions):
                loss_ij = self.loss(masked_action_pred[:, j], masked_action[:, j])
                action_loss += loss_ij * self.branch_weights[i] * self.action_weights[j]

        # Speed loss
        speed_loss = torch.zeros_like(action_loss)
        speed_loss = self.loss(outputs['pred_speed'], supervisions['speed']) * self.speed_weight

        losses = {}
        losses['action_loss'], losses['speed_loss'] = action_loss, speed_loss

        return losses
    
    @staticmethod
    def _get_branch_masks(commands, n_branches, n_actions):
        controls_masks = []

        for i in range(n_branches):
            mask = (commands == i).float()
            mask = torch.cat([mask] * n_actions, 1)
            controls_masks.append(mask)

        return controls_masks

    def sum_losses(self, losses):
        loss = losses['action_loss'] + losses['speed_loss']
        return loss

def make_optimizer(model, lr, wd):
    return torch.optim.Adam(model.parameters(), lr, weight_decay=wd)

def make_scheduler(optimizer, step_size, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

