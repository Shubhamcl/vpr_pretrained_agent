import logging
from os.path import join
import sched

from tqdm.auto import tqdm
import torch

log = logging.getLogger(__name__)

def trainer(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, device, writer, \
        model_dir, save_every=10, epochs=10):
    
    model = model.to(device)

    log.info("Training")

    for epoch in range(epochs):
        
        # Progress bar, loss holders
        progress_bar = tqdm(total=len(train_dataloader), smoothing=0)
        running_loss_train, running_loss_val = 0.0, 0.0
        a_loss_train, s_loss_train = 0.0, 0.0
        a_loss_val, s_loss_val = 0.0, 0.0
        total_train_loss, total_val_loss = 0.0, 0.0

        # Train
        model.train()
        for batch_num, (command, policy_input, supervision) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Send to device
            policy_input = dict([(k, v.to(device)) for k, v in policy_input.items()])
            supervision = dict([(k, v.to(device)) for k, v in supervision.items()])
            command = command.to(device)
            
            output = model(policy_input)
            losses = criterion.forward(output, supervision, command)

            loss = criterion.sum_losses(losses)
            action_loss, speed_loss = losses['action_loss'], losses['speed_loss']
            
            loss.backward()
            optimizer.step()

            # Summing to log later
            running_loss_train += loss.item()
            
            a_loss_train += action_loss.item()
            s_loss_train += speed_loss.item()
            total_train_loss += action_loss.item() + speed_loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch+1} | train loss: {(running_loss_train / (batch_num + 1)):.4f}')

        progress_bar = tqdm(total=len(val_dataloader), smoothing=0)
        
        # Validate
        model.eval()
        for batch_num, (command, policy_input, supervision) in enumerate(val_dataloader):
            # Send to device
            policy_input = dict([(k, v.to(device)) for k, v in policy_input.items()])
            supervision = dict([(k, v.to(device)) for k, v in supervision.items()])
            command = command.to(device)
            
            output = model(policy_input)
            losses = criterion.forward(output, supervision, command, policy_input)            
            
            loss = criterion.sum_losses(losses)
            action_loss, speed_loss = losses['action_loss'], losses['speed_loss']

            # Summing to log later
            running_loss_val += loss.item()

            a_loss_val += action_loss.item()
            s_loss_val += speed_loss.item()
            total_val_loss += action_loss.item() + speed_loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch+1} | validation loss: {(running_loss_val / (batch_num + 1)):.4f}')

        scheduler.step()
        
        # Normalizing stats
        total_loss_train = running_loss_train / len(train_dataloader)
        total_loss_val = running_loss_val / len(val_dataloader)

        log.info(f"epoch {epoch+1} | TrainLoss = {total_loss_train} | ValLoss = {total_loss_val}")

        a_loss_train /= len(train_dataloader)
        s_loss_train /= len(train_dataloader)
        total_train_loss /= len(train_dataloader)

        a_loss_val /= len(val_dataloader)
        s_loss_val /= len(val_dataloader)
        total_val_loss /= len(val_dataloader)

        # Model saving (Weak logic now, save last epoch)
        # if epoch +1 == epochs:
        if (epoch+1) % save_every == 0:
            torch.save(model, join(model_dir, f'model_{epoch+1}.t7'))

        # Log to Tensorboard
        writer.add_scalar('Loss/train', total_loss_train, epoch+1)
        writer.add_scalar('Loss/val', total_loss_val, epoch+1)

        writer.add_scalar('ActionLoss/train', a_loss_train, epoch+1)
        writer.add_scalar('ActionLoss/val', a_loss_val, epoch+1)
        writer.add_scalar('SpeedLoss/train', s_loss_train, epoch+1)
        writer.add_scalar('SpeedLoss/val', s_loss_val, epoch+1)
        writer.add_scalar('SimpleLoss/train', total_train_loss, epoch+1)
        writer.add_scalar('SimpleLoss/val', total_val_loss, epoch+1)

        # break

    writer.close()