import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math as mathe



class ECGNet(nn.Module):
    
    def __init__(self, num_classes = 4,
                 conv1_channels = 64,
                 conv1_kernel =3,
                 conv1_padding =1,
                 conv2_channels = 32,
                 conv2_kernel = 3,
                 conv2_padding = 1,
                 lst_hidden_size = 128,
                 lstm_num_layers = 1,
                 signal_length = None,
                 n_fft = 512,
                 hop_length = 256,
                 device=None):
        
        super(ECGNet, self).__init__()
        
        self.device = device or torch.device("cpu")
        
        self.n_fft = n_fft
        
        self.hop_length = hop_length
        
        self.freq_bins = n_fft // 2 +1
        
        self.time_frames = (signal_length - n_fft) // hop_length +1
        
        freq1 = self._conv_output_size(self.freq_bins, conv1_kernel, conv1_padding)
        
        time1 = self._conv_output_size(self.time_frames, conv1_kernel, conv1_padding)
        
        freq1 //= 2  # MaxPool2d(2,2)
        
        time1 //= 2
        
        freq2 = self._conv_output_size(freq1, conv2_kernel, conv2_padding)
        
        time2 = self._conv_output_size(time1, conv2_kernel, conv2_padding)
        freq2 //= 2  # MaxPool2d(2,2)
        time2 //= 2
        
        self.final_freq = freq2
        self.final_time = time2
        self.final_channels = conv2_channels
        
        rnn_input_size = conv2_channels * freq2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,conv1_channels, kernel_size= (3,3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_channels,conv2_channels, kernel_size=(3,3) , padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2))
                                   )
        
        self.rnn = nn.LSTM(
            input_size = rnn_input_size,
            hidden_size = lst_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True
        )
        
        self.fc = nn.Linear(lst_hidden_size, num_classes)
        
    def _conv_output_size(self, input_size, kernel_size, padding, stride=1):
        return mathe.floor((input_size + 2 * padding - kernel_size) / stride + 1)
        
    def forward(self, x, lengths):
        
        B = x.size(0)
        
        window = torch.hann_window(self.n_fft, device=x.device)
        
        x = torch.stft(x, n_fft=self.n_fft, hop_length = self.hop_length,window=window,
    center=False,  return_complex = True)
        
        
        
        x = x.abs()
        
        valid_frames = (lengths - self.n_fft) // self.hop_length + 1
        
        mask = torch.arange(x.size(2), device=x.device)[None, :] < valid_frames[:, None]
        
        mask = mask.unsqueeze(1)
        
        spectoogram = x * mask.float()
        
        spectoogram = spectoogram.unsqueeze(1)
        
        x = self.conv1(spectoogram)
        
        x = self.conv2(x)
        
        x = x.permute(0, 3, 1, 2)  # [B, T', C, F']
        
        x = x.reshape(B, x.size(1), -1)  # [B, T', C * F']
        
        lengths_conv1 = (valid_frames + 1) // 2  # Tras conv1
        lengths_conv2 = (lengths_conv1 + 1) // 2  # Tras conv2
        rnn_lengths = lengths_conv2.clamp(min=1, max=x.size(1))
        

        # # # Packed RNN
        packed = pack_padded_sequence(x, rnn_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.rnn(packed)

        out = self.fc(ht[-1])  # [B, num_classes]
        return out
