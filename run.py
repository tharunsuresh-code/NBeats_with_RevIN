import warnings

import os
import argparse
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.nn import functional as F
from torchmetrics import MeanAbsolutePercentageError
from torchinfo import summary

from nbeats_pytorch.model import NBeatsNet
from model.RevIN import RevIN
from model.NBeatsRevin import NBeatsRevin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

pd.set_option('display.float_format', '{:.4f}'.format)
warnings.filterwarnings(action='ignore', message='Setting attributes')

torch.manual_seed(42)

# simple batcher.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch

def main(args):
    data_path = args['data_path']
    forecast_length = args['forecast_length']
    backcast_length = 8 * forecast_length
    batch_size = args['batch_size']
    n_epochs = args['n_epochs']
    out_path = args['out_path']
    results = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE'])
    count = 1

    for csv in os.listdir(data_path):
        file_path = os.path.join(data_path, csv)
        file_name = csv.split(".")[0]
        ts_values = pd.read_csv(file_path, index_col=0, usecols=[0,1])
        ts_values = ts_values.values.flatten()  # just keep np array here for simplicity.

        # data backcast/forecast generation.
        x, y = [], []
        for epoch in range(backcast_length, len(ts_values) - forecast_length):
            x.append(ts_values[epoch - backcast_length:epoch])
            y.append(ts_values[epoch:epoch + forecast_length])
        x = np.array(x)
        y = np.array(y)

        # split train/test.
        c = int(len(x) * 0.75)
        x_train, y_train = x[:c], y[:c]
        x_test, y_test = x[c:], y[c:]

        # model
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK)
        hidden_layer_units=128
        net = NBeatsRevin(stack_types, forecast_length, backcast_length, hidden_layer_units, device=DEVICE)
        net.to(DEVICE)
        optimiser = optim.Adam(lr=1e-3, params=net.parameters())
        criterion = MeanAbsolutePercentageError().to(DEVICE)
        other_criterion = {'mse': torch.nn.MSELoss().to(DEVICE),
                            'mae': torch.nn.L1Loss().to(DEVICE)}

        grad_step = 0
        for epoch in tqdm(range(n_epochs)):
            # train.
            net.train()
            train_loss = []
            for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
                grad_step += 1
                optimiser.zero_grad()
                _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float, device=DEVICE))
                loss = criterion(forecast, torch.tensor(y_train_batch, dtype=torch.float, device=DEVICE))
                train_loss.append(loss.item())
                #loss.requires_grad = True
                loss.backward()
                optimiser.step()
            train_loss = np.mean(train_loss)

            if (epoch+1) % 500 == 0:
                if args['save']:    
                    with torch.no_grad():
                        torch.save(net, os.path.join(args['save_path'], 'nbeatsrevin_' + file_name + ".pth"))

        #test
        net.eval()
        test_inputs = ts_values[c-backcast_length:c].tolist()
        for i in range(c, len(ts_values), forecast_length):
            seq = torch.FloatTensor([test_inputs[-backcast_length:]]).to(DEVICE)
            with torch.no_grad():
                _, forecast = net(seq)
                test_inputs.extend(forecast.detach().cpu().numpy().tolist()[0])

        start_range = len(ts_values) - len(test_inputs) + backcast_length
        
        test_outputs = torch.tensor(test_inputs[backcast_length:], dtype=torch.float, device=DEVICE)
        true_outputs = torch.tensor(ts_values[start_range:], dtype=torch.float, device=DEVICE)
        mape_loss = criterion(test_outputs, true_outputs).item()
        mae_loss = other_criterion['mae'](test_outputs, true_outputs).item()
        mse_loss = other_criterion['mse'](test_outputs, true_outputs)
        rmse_loss = torch.sqrt(mse_loss).item()
        mse_loss = mse_loss.item()

        #saving test scores
        results.loc[file_name] = [mae_loss*1000,
                                        mse_loss*1000,
                                        rmse_loss*1000,
                                        mape_loss*1000]

        print(f'Filename: {csv}, Progress - {count}/{len(os.listdir(data_path))}\n'
                f'epoch = {str(epoch).zfill(4)}, '
                f'grad_step = {str(grad_step).zfill(6)}, '
                f'tr_loss (mape) = {1000 * train_loss:.3f}, '
                f'te_loss (mape) = {1000 * mape_loss:.3f}')
        count+=1

        #saving prediction plot
        x_axis = np.arange(start_range, len(ts_values))
        plt.title(csv)
        plt.xlabel('timestamps')
        plt.ylabel('y-value')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(range(x_axis[0]), ts_values[:x_axis[0]])
        plt.plot(x_axis, ts_values[x_axis[0]:], linestyle='--', alpha=0.3, color='b')
        plt.plot(x_axis,np.array(test_inputs[backcast_length:]), linestyle='--', alpha=0.5, color='r')
        plt.vlines(x = x_axis[0], ymin = min(ts_values), ymax = max(ts_values), 
                colors = 'purple', 
                label = 'vline_multiple - full height') 
        plt.savefig(os.path.join(out_path, file_name + ".png"))
        plt.clf()
        #plt.show()

    #saving test scores to csv file
    results.index.name = 'Filename'
    results.index = results.index.astype(int)
    results.sort_index().round(4).to_csv(os.path.join(out_path, 'test_scores.csv'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply NBeats on time-series data')
    parser.add_argument('-forecast_length', default=6, type=int)
    parser.add_argument('-data_path', default='dataset',
                        help='location of dataset folder containing the time series files in csv format')
    parser.add_argument('-out_path', default='output',
                        help='save location for the output plots and scores on test split (in a csv)')
    parser.add_argument('-save_path', default='save',
                        help='save location for the trained NBeatsRevin model')
    parser.add_argument('-random_seed', default=42, type=int,
                        help='Random seed for torch')
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-n_epochs', default=1500, type=int)
    parser.add_argument('-save', action='store_true')
    args = vars(parser.parse_args())
    main(args)