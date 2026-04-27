import torch
import torch.nn as nn

class PointPredictorMLP(nn.Module):
    """
    A lightweight Multi-Layer Perceptron (MLP) for predicting 3D point coordinates.
    
    This model takes a flattened vector of 'k' previous points (context) and 
    predicts the (x, y, z) coordinates of the next point (target).
    
    Args:
        context_size (int): The number of neighbor points to use as input. Default is 5.
        hidden_dim (int): The number of neurons in each hidden layer. Default is 64.
    """
    def __init__(self, context_size=5, hidden_dim=64):
        super(PointPredictorMLP, self).__init__()
        
        input_dim = context_size * 3
        
        output_dim = 3
        
        self.network = nn.Sequential(
            # Layer 1: Input -> Hidden
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            
            # Layer 2: Hidden -> Hidden
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # Layer 3: Hidden -> Hidden
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Prediction of shape (batch_size, 3).
        """
        return self.network(x)

if __name__ == "__main__":
    model = PointPredictorMLP(context_size=5)
    print("Model Architecture:\n", model)
    
    fake_input = torch.randn(2, 15) 
    output = model(fake_input)
    
    print("\nTest Input Shape:", fake_input.shape)
    print("Test Output Shape:", output.shape)
    print("Model implementation is valid!")