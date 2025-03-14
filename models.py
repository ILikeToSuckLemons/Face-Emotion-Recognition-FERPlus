import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartModel(nn.Module):
    def __init__(self, input_shape=(1, 48, 48), leaky_relu_slope=0.2, dropout_rate=0.5, regularization_rate=0.0001, n_classes=8, logits=False):
        super(SmartModel, self).__init__()
        
        self.logits = logits
        self.n_classes = n_classes
        self.input_shape = input_shape

        # Convolutional layers with batch normalization, max pooling, and dropout
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(dropout_rate)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn8(self.conv8(x)), negative_slope=0.2)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layer
        x = self.fc(x)

        if not self.logits:
            x = F.softmax(x, dim=1)

        return x

# BaseModel inherits from SmartModel without changing the architecture
class BaseModel(SmartModel):
    def __init__(self, input_shape=(1, 48, 48), leaky_relu_slope=0.2, dropout_rate=0.5, regularization_rate=0.0001, n_classes=8, logits=False):
        super(BaseModel, self).__init__(input_shape, leaky_relu_slope, dropout_rate, regularization_rate, n_classes, logits)

# PerformanceModel also inherits from SmartModel
class PerformanceModel(SmartModel):
    def __init__(self, input_shape=(1, 48, 48), leaky_relu_slope=0.2, dropout_rate=0.5, regularization_rate=0.0001, n_classes=8, logits=False):
        super(PerformanceModel, self).__init__(input_shape, leaky_relu_slope, dropout_rate, regularization_rate, n_classes, logits)

# Function to easily create an instance of the PerformanceModel
def get_performance_model(input_shape=(1, 48, 48), leaky_relu_slope=0.2, dropout_rate=0.5, regularization_rate=0.0001, n_classes=8, logits=False):
    return PerformanceModel(input_shape=input_shape,
                           leaky_relu_slope=leaky_relu_slope,
                           dropout_rate=dropout_rate,
                           regularization_rate=regularization_rate,
                           n_classes=n_classes,
                           logits=logits)