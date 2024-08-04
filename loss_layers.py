import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """
    Content loss module for neural style transfer.

    Parameters:
        target (torch.Tensor): The target content feature map.

    Attributes:
        loss (torch.Tensor): Content loss value.

    """

    def __init__(self,target=None):
        super(ContentLoss, self).__init__()
        self.loss = None
        if target is not None:
            self.target = target

    def forward(self, input_feature_map):
        """
        Calculate the loss between the input image feature map and the target content feature map.
        Return the input image feature map to maintain the continuity of the model flow.

        Args:
            input_feature_map (torch.Tensor): The input image feature map.

        Returns:
            torch.Tensor: The input image feature map.
        """
        if self.target is not None:
            self.loss = F.mse_loss(input_feature_map, self.target)
        return input_feature_map

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """
    Style loss module for neural style transfer.

    Parameters:
        target_feature (torch.Tensor): The target style feature map.

    Attributes:
        loss (torch.Tensor): Style loss value.
        target (torch.Tensor): Gram matrix of the target style feature map.

    """

    def __init__(self,target_feature_map=None):
        super(StyleLoss, self).__init__()
        self.loss = None
        if target_feature_map is not None:
            self.target = gram_matrix(target_feature_map)


    def forward(self, input_feature_map):
        """
        Calculate the loss between the Gram matrix of the input image feature map and the target style Gram matrix.
        Return the input image feature map to maintain the continuity of the model flow.

        Args:
            input_feature_map (torch.Tensor): The input image feature map.

        Returns:
            torch.Tensor: The input image feature map.
        """
        if self.target is not None:
            G = gram_matrix(input_feature_map)
            self.loss = F.mse_loss(G, self.target)
        return input_feature_map