import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) * dilation,
                      dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) * dilation,
                      dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection with cropping
        if out.size(-1) > x.size(-1):
            out = out[..., :x.size(-1)]
        elif out.size(-1) < x.size(-1):
            x = x[..., :out.size(-1)]

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCN_STFT_Classifier(nn.Module):
    def __init__(self, num_classes=None, n_fft=None, hop_length=None,
                 num_levels=None, kernel_size=None, dropout=None,
                 hidden_channels=None, device=None):  # Añadir parámetro device
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Gestión de dispositivo

        layers = []
        in_ch = self.freq_bins
        for i in range(num_levels):
            dilation = 2 ** i
            out_ch = hidden_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers).to(self.device)
        self.global_pool = nn.AdaptiveAvgPool1d(1).to(self.device)
        self.fc = nn.Linear(in_ch, num_classes).to(self.device)

    def forward(self, x, lengths=None):
        # Asegurar que la entrada esté en el dispositivo correcto
        x = x.to(self.device)
        B = x.size(0)
        
        # Crear ventana en el dispositivo correcto
        window = torch.hann_window(self.n_fft, device=self.device)
        
        # Calcular STFT
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=False
        )
        x = x.abs()  # Magnitud
        x = x.to(dtype=torch.float32)  # Asegurar tipo correcto

        # Manejar padding con máscara
        if lengths is not None:
            lengths = lengths.to(self.device)  # Asegurar que lengths esté en el dispositivo correcto
            valid_frames = (lengths - self.n_fft) // self.hop_length + 1
            max_frames = x.size(2)
            
            # Crear máscara en el dispositivo correcto
            mask = torch.arange(max_frames, device=self.device)[None, :] < valid_frames[:, None]
            mask = mask.unsqueeze(1).float()  # (B, 1, T')
            x = x * mask

        # Pasar a través de la TCN
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(2)
        x = self.fc(x)
        return x