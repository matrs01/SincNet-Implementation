from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import math
import soundfile as sf


class SincNet_Dataset(Dataset):
    def __init__(self, batch_size: int, data_path: str, track_list: list, 
                       num_tracks: int, input_dim: int ,labels_dict: dict):
        super(SincNet_Dataset, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.track_list = track_list
        self.num_tracks = num_tracks
        self.input_dim = input_dim
        self.labels_dict = labels_dict
        self.amplitude = 0.2
    
    def __len__(self) -> int:
        return self.num_tracks

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track, _ = sf.read(self.data_path+self.track_list[item])
        # crop a random chunk
        chunk_start = np.random.randint(track.shape[0]-self.input_dim-1)

        if len(track.shape) == 2:
            track = track[:,0]
        
        return torch.tensor(track[chunk_start: chunk_start+self.input_dim], dtype=torch.float32), \
                torch.tensor(self.labels_dict[self.track_list[item]])



class Valid_DataLoader():
    def __init__(self, batch_size: int, data_path: str, track_list: list, 
                       num_tracks: int, input_dim: int , shift_size: int,
                        labels_dict: dict):
        self.batch_size = batch_size
        self.data_path = data_path
        self.track_list = track_list
        self.num_tracks = num_tracks
        self.input_dim = input_dim
        self.shift_size = shift_size
        self.labels_dict = labels_dict
        self.last_returned_track = 0
        self.current_track = None
        self.current_target = None
        self.current_pos = 0

    def __iter__(self):
        self.current_track, _ = sf.read(self.data_path+self.track_list[0])
        self.current_target = self.labels_dict[self.track_list[0]]
        self.last_returned_track = 0
        self.current_pos = 0
        return self
    
    def __next__(self) -> Tuple[torch.tensor, torch.tensor, int]:
        # remaining_len = math.ceil((self.current_track.shape[0] - self.input_dim -\
        #                            self.current_pos + 1) / self.shift_size)
        remaining_len = math.floor((self.current_track.shape[0] - self.current_pos -\
                                   self.input_dim) / self.shift_size + 1)
        if remaining_len == 0:
            self.last_returned_track += 1
            if self.last_returned_track == self.num_tracks:
                raise StopIteration

            self.current_track, _ = sf.read(self.data_path + 
                                            self.track_list[self.last_returned_track])
            self.current_target = self.labels_dict[self.track_list[self.last_returned_track]]
            self.current_pos = 0
            remaining_len = math.floor((self.current_track.shape[0] - self.current_pos -\
                                   self.input_dim) / self.shift_size + 1)

        cur_batch_size = self.batch_size if remaining_len >= self.batch_size else remaining_len

        X = torch.zeros((cur_batch_size, self.input_dim))
        y = torch.zeros((cur_batch_size), dtype=torch.long) + self.current_target

        for i in range(cur_batch_size):
            X[i] = torch.tensor(self.current_track[self.current_pos : self.current_pos+self.input_dim])
            self.current_pos += self.shift_size
        
        return X.float(), y, self.last_returned_track


def train(model: nn.Module, train_dataloader: DataLoader, valid_dataloader: Valid_DataLoader, 
          optimizer, criterion, num_epochs: int, 
          N_eval_epochs: int, save_path: str) -> Tuple[list, list, list, list, list]:
    train_losses = []
    train_cerr = []
    val_losses = []
    val_cerr = []
    val_sentece_cerr = []
    best_val_sentece_cerr = 1.0
    # buffer for valid sentences classification
    probs_buffer = np.zeros(shape=(valid_dataloader.num_tracks, model.num_classes))
    gt_val_classes = np.zeros(shape=(valid_dataloader.num_tracks))
    for epoch in range(num_epochs):
        epoch_train_losses = []
        epoch_train_cerr = []
        epoch_val_losses = []
        epoch_val_cerr = []

        model.train()
        for X, y in train_dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.detach().cpu().numpy())
            epoch_train_cerr.append(np.mean((torch.argmax(
                                preds.detach(), dim=-1) != y).cpu().numpy()))
        train_losses.append(sum(epoch_train_losses) / len(train_dataloader))
        train_cerr.append(sum(epoch_train_cerr) / len(train_dataloader))
        # print(f'(epoch {epoch}) TRAIN: loss = {train_losses[-1]}, cerr = {train_cerr[-1]}')


        if epoch % N_eval_epochs == 0:
            model.eval()
            n = 0 # len of valid_dataloader
            for X, y, track_i in valid_dataloader:
                n += 1
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                preds = model(X).detach()
                loss = criterion(preds, y)
                epoch_val_losses.append(loss.detach().cpu().numpy())
                epoch_val_cerr.append(np.mean((torch.argmax(
                                preds.detach(), dim=-1) != y).cpu().numpy()))
                
                gt_val_classes[track_i] = y[0].cpu().numpy()
                probs = torch.exp(preds) / torch.exp(preds).sum(-1).unsqueeze(-1)
                probs_buffer[track_i] += probs.sum(0).cpu().numpy()

            val_losses.append(sum(epoch_val_losses) / n)
            val_cerr.append(sum(epoch_val_cerr) / n)

            sentence_cerr = np.mean(np.argmax(probs_buffer, axis=-1) != gt_val_classes)
            val_sentece_cerr.append(sentence_cerr)

            probs_buffer = np.zeros(shape=(valid_dataloader.num_tracks, model.num_classes))
            gt_val_classes = np.zeros(shape=(valid_dataloader.num_tracks))

            if val_sentece_cerr[-1] < best_val_sentece_cerr:
                best_val_sentece_cerr = val_sentece_cerr[-1]
                torch.save(model.state_dict(), 'save_path')
            if epoch != 0:    
                print(f'(epoch {epoch}) TRAIN: loss = {sum(train_losses[-N_eval_epochs:])/N_eval_epochs}, cerr = {sum(train_cerr[-N_eval_epochs:])/N_eval_epochs}')
            print(f'(epoch {epoch}) VALID: loss = {val_losses[-1]}, cerr = {val_cerr[-1]}, snt_cerr = {val_sentece_cerr[-1]}')
        
    
    return train_losses, train_cerr, val_losses, val_cerr, val_sentece_cerr

def ReadList(list_file):
    f = open(list_file,"r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig