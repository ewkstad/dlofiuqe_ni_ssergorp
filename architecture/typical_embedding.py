import torch

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, n_c, l_max, n_t, device):
        super(EmbeddingLayer, self).__init__()
        self.n_t = n_t
        self.l_max = l_max
        self.n_c = n_c
        self.C = torch.nn.Parameter(torch.randn(n_t, n_c * ((l_max + 1) ** 2)))
        self.device = device
    
    def forward(self, input):
        # Ensure input is a tensor
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
    
        # Check input values are in valid range
        if input.min() < 0 or input.max() > self.n_t - 1:
            raise ValueError('Invalid input values')
      
        # Convert input tensor to long tensor
        input = input.long()
    
        # Transfer input tensor to correct device
        input = input.to(self.device)
    
        return self.C[input]