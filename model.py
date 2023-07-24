import torch
import torch.nn as nn
from torchvision.models.squeezenet import SqueezeNet
from torchvision.models import resnet34


seg_vpr_address = ""
import sys
sys.path.insert(0, seg_vpr_address)

def segvpr_encoder(pre_trained_address):
    encoder = torch.load(pre_trained_address)
    segvpr_encoder = nn.Sequential(*[
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
        nn.AdaptiveAvgPool2d(output_size=(1,1)),
        nn.Flatten()
    ])
    return segvpr_encoder


class RoachNet(nn.Module):
    def __init__(self, n_input_channels=3, n_action_outputs=2, n_branches=6,
        wide_image=False, lb_mode=False):
        super(RoachNet, self).__init__()
        
        if lb_mode:
            num_of_measurements = 9
        else:
            num_of_measurements = 1

        action_output_dim = 2

        # Perception block
        self.perception = resnet34(pretrained=True)

        self.measurements = nn.Sequential(
            nn.Linear(num_of_measurements, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.join = nn.Sequential(
            nn.Linear(1000+128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.action_branches = nn.ModuleList([
            nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, action_output_dim)
        ) for i in range(n_branches)
        ]) 

    def forward(self, obs):
        x = self.perception(obs['im'])

        # Speed prediction
        outputs = {'pred_speed': self.speed_branch(x)}

        # Action prediction
        m = self.measurements(obs['state'])
        j = torch.cat([x, m], dim=1)
        j = self.join(j)
        
        # a = torch.cat([out(j) for out in self.conditional_branches], dim=1)
        a = [out(j) for out in self.action_branches]
        outputs['action_branches'] = a

        return outputs

class RoachVPR(nn.Module):
    def __init__(self, n_input_channels=3, n_action_outputs=2, n_branches=6, 
                 pre_trained_address="", wide_image=False, lb_mode=False):
        super(RoachVPR, self).__init__()
        
        if lb_mode:
            num_of_measurements = 9
        else:
            num_of_measurements = 1

        action_output_dim = 2

        # Perception block
        self.perception = segvpr_encoder(pre_trained_address)

        self.measurements = nn.Sequential(
            nn.Linear(num_of_measurements, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.join = nn.Sequential(
            nn.Linear(2048+128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.action_branches = nn.ModuleList([
            nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, action_output_dim)
        ) for i in range(n_branches)
        ]) 

    def forward(self, obs):
        x = self.perception(obs['im'])

        # Speed prediction
        outputs = {'pred_speed': self.speed_branch(x)}

        # Action prediction
        m = self.measurements(obs['state'])
        j = torch.cat([x, m], dim=1)
        j = self.join(j)
        
        # a = torch.cat([out(j) for out in self.conditional_branches], dim=1)
        a = [out(j) for out in self.action_branches]
        outputs['action_branches'] = a

        return outputs
