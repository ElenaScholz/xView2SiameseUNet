import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNet_ResNet50(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Load ResNet50 als Encoder
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Encoder: Uses the inital layers of ResNet50 to extract the features
        self.encoder = nn.Sequential(
            resnet.conv1, # Initial convolutional layer
            resnet.bn1, # Batch Normalization
            resnet.relu, # Activation function
            resnet.maxpool, # Downsampling trough max pooling
            # Residual blocks
            resnet.layer1, 
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Decoder: Upsample the features to regain spatial resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Final classification layer: Reduces channel dimension to the number of classes
        self.final_conv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        """
        Defines the forward pass through the network.
        
        Args:
            x (tensor): Input tensor with shape [batch_size, channels, height, width].

        Returns:
            out (tensor): Output tensor after segmentation with shape matching the input's spatial dimensions.
        """

        # Save the original input spatial dimensions for later upsampling

        original_size = x.shape[2:]
        
        try:
            # Pass the input through encoder
            features = self.encoder(x)
           
            # Pass the extracted features through the decoder to upsampple
            decoder_out = self.decoder(features)
            
            # Apply the final convolution to predict segmentation classes
            out = self.final_conv(decoder_out)


            # # Upsample the final output to match the original input size
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

            return out

        except Exception as e:
            # Print and raise error if any issues occur during the forward pass
            print(f"Error in forward pass: {e}")
            raise
