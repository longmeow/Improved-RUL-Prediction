import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import TimeSeriesDataset
from trainer import ModelTrainer
from model import create_transformer, create_fnet_hybrid
from utils import save_config, get_args, create_dirs, process_config

torch.manual_seed(42)


def main():
    start = time.perf_counter()

    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("ERROR: Missing or invalid config file.")
        sys.exit(1)

    create_dirs(config["result_dir"], config["model_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TimeSeriesDataset(config, mode='train')
    train_loader = DataLoader(train_data,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'])

    if config['model'] == 1:
        model = create_transformer(N=config['num_layers'],
                                        d_model=config['d_model'],
                                        l_win=config['l_win'],
                                        device=None,
                                        kernel_size=config['kernel_size'],
                                        d_ff=config['dff'],
                                        h=config['n_head'],
                                        dropout=config['dropout'])
    if config['model'] == 2:
        model = create_fnet_hybrid(N=config['num_layers'],
                                        d_model=config['d_model'],
                                        l_win=config['l_win'],
                                        device=None,
                                        kernel_size=config['kernel_size'],
                                        d_ff=config['dff'],
                                        h=config['n_head'],
                                        dropout=config['dropout'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, config)

    trainer.train()
    config = trainer.update_config()

    save_config(config['result_dir'] + "result_lr_{}_l_win_{}_dff_{}.yml".format(
        config['lr'], config['l_win'], config['dff']), config)

    print('DONE.')
    total = (time.perf_counter() - start) / 60
    print('Training time: {}'.format(total))

if __name__ == "__main__":
    main()
