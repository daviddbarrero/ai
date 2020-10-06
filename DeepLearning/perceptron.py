import torch

class SLP(torch.nn.Module):
    " Neuron of on layer to aproxime functions"

    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        # Shape or size of input or output

        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape) # Relu activation function
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)



    def forward(self, x):
        x = torch.form_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x
