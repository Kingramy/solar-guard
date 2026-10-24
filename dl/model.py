# dl/model.py
import torch
import torch.nn as nn

def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    A foundational building block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU.
    Using padding=1 keeps the spatial dimensions identical before pooling.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ThermalCNN(nn.Module):
    def __init__(self, n_classes: int = 12):
        super().__init__()
        # Input tensor shape: (Batch_Size, 1_Channel, 40_Height, 24_Width)
        self.features = nn.Sequential(
            conv_block(1, 32),
            nn.MaxPool2d(2),      # Output spatial shape: 20 x 12
            
            conv_block(32, 64),
            nn.MaxPool2d(2),      # Output spatial shape: 10 x 6
            
            conv_block(64, 128),  # Output spatial shape: 10 x 6
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Crushes 10x6 down to 1x1
            nn.Flatten(),            # Flattens (Batch, 128, 1, 1) -> (Batch, 128)
            nn.Dropout(0.30),        # Randomly zeroes out 30% of neurons to prevent overfitting
            nn.Linear(128, n_classes), # Maps the 128 features to 12 raw class scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The forward pass defining how data moves through the network
        return self.head(self.features(x))

if __name__ == "__main__":
    # Sanity check: Pass a dummy tensor through to verify the math
    model = ThermalCNN()
    dummy_input = torch.zeros(4, 1, 40, 24) # Batch of 4, 1 channel, 40x24
    output = model(dummy_input)
    print(f"Model built successfully. Output shape: {output.shape}") 
    # Should print (4, 12) -> 4 images, 12 class scores each