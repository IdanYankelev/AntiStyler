import torch
import torch.nn as nn
import torch.optim as optim
from loss_layers import StyleLoss, ContentLoss

class antistyle:
    """
    Class for performing anti-style transfer.

    Parameters:
        cnn (torch.nn.Module): The base feature extraction convolutional neural network.
        content_layers (list): List of layer names to be considered for content loss.
        style_layers (list): List of layer names to be considered for style loss.
        content_weight (float): Weight for the content loss.
        style_weight (float): Weight for the style loss.

    Attributes:
        content_layers (list): List of layer names for content loss.
        style_layers (list): List of layer names for style loss.
        content_weight (float): Weight for the content loss.
        style_weight (float): Weight for the style loss.
        model (torch.nn.Module): anti-style transfer model.
        style_losses (list): List of style losses.
        content_losses (list): List of content losses.

    """

    def __init__(self,cnn, content_layers, style_layers, content_weight, style_weight,num_steps):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps
        self.model = None
        self.style_losses = None
        self.content_losses = None
        self.set_base_model(cnn)

    def set_base_model(self,cnn):
        """
        Sets up the base model by trimming unnecessary layers and adding content and style loss layers.

        Args:
            cnn (torch.nn.Module): The base feature extraction convolutional neural network.

        Raises:
            ValueError: If no ContentLoss or StyleLoss layers are found in the model.

        """
        model = nn.Sequential()

        i = 0  # increment every time we see a conv layer.
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # add hidden content loss layer after content layer.
            if name in self.content_layers:
                model.add_module("content_loss_{}".format(i), ContentLoss())

            # add hidden style loss layer after style layer.
            if name in self.style_layers:
                model.add_module("style_loss_{}".format(i), StyleLoss())

        # Find the index of the last content or style loss layer.
        last_loss_layer_index = None
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss)):
                last_loss_layer_index = i
                break

        # Check if any content or style loss layers were found.
        if last_loss_layer_index is not None:
            # Trim off layers after the last content or style loss layer.
            trimmed_model = model[:last_loss_layer_index + 1]
            self.model = trimmed_model
        else:
            # Handle the case where no content or style loss layers were found.
            raise ValueError("No ContentLoss or StyleLoss layers found in the model.")

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.requires_grad_(False)


    def update_targets(self,content_image, style_image):
        """
        Update the target feature maps for content and style images.

        Args:
            content_image (torch.Tensor): Content image.
            style_image (torch.Tensor): Style image.

        """
        content_losses = []
        style_losses = []

        content_image_tensor = content_image.detach()
        style_img_tensor = style_image.detach()

        i = 0  # increment every time we see a conv layer.
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss):
                continue
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            content_image_tensor = layer(content_image_tensor).detach()
            style_img_tensor = layer(style_img_tensor).detach()

            # update hidden content loss layers:
            if name in self.content_layers:
                content_loss = ContentLoss(content_image_tensor)
                setattr(self.model, f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)

            # update hidden style loss layers:
            if name in self.style_layers:
                style_loss = StyleLoss(style_img_tensor)
                setattr(self.model, f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

        self.style_losses = style_losses
        self.content_losses = content_losses

    def apply(self,content_image, style_image, input_img):
        """
        Apply the anti-style transfer to the input image.

        Args:
            content_image (torch.Tensor): Content image.
            style_image (torch.Tensor): Style image.
            input_img (torch.Tensor): Input image to be optimized.

        Returns:
            torch.Tensor: Optimized input image.

        """
        self.update_targets(content_image,style_image)

        input_img.requires_grad_(True)

        optimizer = optim.Adam([input_img]) #LBFGS([input_img],max_iter=1)

        def anti_style_step():
            optimizer.zero_grad()
            self.model(input_img)
            style_loss = -sum(sl.loss for sl in self.style_losses)
            style_loss.backward()
            return style_loss

        def content_step():
            optimizer.zero_grad()
            self.model(input_img)
            content_loss = sum(cl.loss for cl in self.content_losses)
            style_loss = -sum(sl.loss for sl in self.style_losses)
            total_loss = content_loss * self.content_weight + style_loss * self.style_weight
            total_loss.backward()
            return total_loss

        # Perform optimization steps.
        for _ in range(0,self.num_steps):
            optimizer.step(anti_style_step)
            optimizer.step(content_step)
            # Clip pixel values to valid range.
            input_img.data.clamp_(0, 1)

        return input_img.detach()