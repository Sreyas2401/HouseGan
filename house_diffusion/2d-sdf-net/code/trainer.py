import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import SDFNet
from loader import SDFData
from renderer import plot_sdf
from gradient import getGradient, getGradientAndHessian
from loss import implicit_loss_2d

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
MODEL_PATH = '../models/'
RES_PATH = '../results/trained_heatmaps/'
MASK_PATH = '../shapes/masks/'
LOG_PATH = '../logs/'

if __name__ == '__main__':
    batch_size = 64
    learning_rate = 1e-5 # Play around ------> :)
    epochs = 100
    regularization = 0  # Default: 1e-2
    delta = 0.1  # Truncated distance

    print('Enter shape name for image without hessian loss:')
    name = input()

    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = SDFNet().to(device)
    if os.path.exists(f'{MODEL_PATH}{name}.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

    writer = SummaryWriter(LOG_PATH)
    total_train_step = 0
    total_val_step = 0
    total_loss = 0
    total_hess_loss = 0
    start_hess_track = False
    start_time = time.time()
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')

        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (xy, sdf) in enumerate(train_dataloader):
            xy, sdf = xy.to(device), sdf.to(device)
            xy.requires_grad = True
            pred_sdf = model(xy)
            sdf = torch.reshape(sdf, pred_sdf.shape)
            loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))

            # Once epoch reaches 1000 and 1200, considering 1200 is the maxepoch, render the image.
            if t > 100:    # Changed this to > 1000 instead of > 500 epochs :)
                start_hess_track = True
                predicted_gradient, pred_hess_matrix = getGradientAndHessian(pred_sdf, xy, matrixsize=2)
                hessloss = implicit_loss_2d(predicted_gradient, pred_hess_matrix, device)
                print(hessloss)
                loss += hessloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            if batch % 50 == 0:
                loss_value, current = loss.item(), batch * len(xy)
                print(f'loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]')

            total_train_step += 1
            if total_train_step % 200 == 0:
                writer.add_scalar('Training loss', loss.item(), total_train_step)

        # Evaluation loop
        model.eval()
        size = len(val_dataloader.dataset)
        val_loss = 0

        with torch.no_grad():
            for xy, sdf in val_dataloader:
                xy, sdf = xy.to(device), sdf.to(device)
                pred_sdf = model(xy)
                sdf = torch.reshape(sdf, pred_sdf.shape)
                loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))
                val_loss += loss

        val_loss /= size
        end_time = time.time()
        print(f'Test Error: \n Avg loss: {val_loss:>8f} \n Time: {(end_time - start_time):>12f} \n ')

        total_val_step += 1
        writer.add_scalar('Val loss', val_loss, total_val_step)

        total_loss += val_loss

        if(start_hess_track):
            total_hess_loss += val_loss

        if t==1000:             # If t == 1000 epochs then it renders image without hess loss function :)
            print('Total loss contributed without hessian loss : ', total_loss)
            print('Rendering image without applying hessian loss')
            plot_sdf(model, device, res_path=RES_PATH, name=name, mask_path=MASK_PATH, is_net=True, show=True)

    torch.save(model.state_dict(), f'{MODEL_PATH}{name}.pth')
    print(f'Complete training with {epochs} epochs!')
    print('Total loss :', total_loss)
    print('Total loss contributed by hessian loss :', total_hess_loss)

    writer.close()

    # Plot results
    print('Plotting results...')
    plot_sdf(model, device, res_path=RES_PATH, name=name, mask_path=MASK_PATH, is_net=True, show=False)
    print('Done!')