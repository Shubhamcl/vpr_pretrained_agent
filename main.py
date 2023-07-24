import hydra
from omegaconf import DictConfig, OmegaConf
from os.path import join

from data import get_dataloader
from model import PilotNetConditional, PilotNetConditionalwJModule, SqueezeNetConditional, SqueezeNetConditionalwJModule, RoachNet, RoachVPR
from train import trainer
from loss import BranchedLoss, make_optimizer, make_scheduler
from utils import select_device, set_seed, make_loggers

# [[ Settings ]] 

@hydra.main(config_path="configs", config_name="main_train")
def main(cfg : DictConfig) -> None:
    
    # Misc
    set_seed()
    model_dir, writer = make_loggers(cfg.log_dir, cfg.exp_name)
    OmegaConf.save(cfg, join(model_dir, '..', 'config.yaml'))

    # [[ Data ]]

    train_loader, val_loader = get_dataloader(cfg.data_address, cfg.dagger_data_address, 
        cfg.batch_size, cfg.wide_image, cfg.lb_mode, cfg.aug)

    # [[ Model ]]

    if cfg.pre_train:
        model = RoachVPR(pre_trained_address= cfg.pre_train_path,lb_mode=cfg.lb_mode)
    else:
        model = RoachNet(lb_mode=cfg.lb_mode)

    # [[ Criterion, Optimizer, Scheduler, Cuda ]]

    criterion = BranchedLoss()
    optimizer = make_optimizer(model, cfg.lr, cfg.wd)
    scheduler = make_scheduler(optimizer, cfg.lr_step_size)
    device = select_device()

    # [[ Trainer ]]

    trainer(train_loader, val_loader, model, criterion, optimizer, scheduler, \
        device, writer, model_dir, cfg.save_every, cfg.epochs)

if __name__=="__main__":
    main()
