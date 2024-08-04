import torch
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models import antistyle_model
from config import *

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'Dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def resize(image,size):
    return transforms.Resize(size, antialias=True)(image)

def load_image(image_path):
    """
    Load an image from the given path, convert it to RGB if it's grayscale, and return it as a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The image as a PyTorch tensor.
    """
    image_path = os.path.join(os.getcwd(), image_path)

    file_type = image_path.split('.')[-1]

    if file_type == 'npy':
        image = np.load(image_path) / 255.0
    else:
        image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    if image_tensor.shape[1]==1:
        image_tensor = torch.stack([image_tensor.squeeze(0)] * 3,dim=1)
    return image_tensor.to(get_device(), torch.float)

def plot_predictions(undefended_image, undefended_predictions, defended_image, defended_predictions):
    def plot_image_predictions(image, predictions, ax, title):
        # Convert the image tensor to a PIL image
        image = image.squeeze(0).cpu()  # remove the batch dimension and move to CPU
        image = transforms.ToPILImage()(image)

        # Convert image to numpy array for plotting
        image_np = np.array(image)

        # Extract boxes, labels, and scores from predictions
        boxes = predictions[0]['boxes'].cpu()
        labels = predictions[0]['labels'].cpu()
        scores = predictions[0]['scores'].cpu()

        ax.imshow(image_np)
        ax.set_title(title)

        # Plot each box
        for i, box in enumerate(boxes):
            if scores[i] >= 0.8:  # Filter out boxes with low confidence
                xmin, ymin, xmax, ymax = box
                width, height = xmax - xmin, ymax - ymin

                # Create a rectangle patch
                rect = plt.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Annotate the box with the label and score
                label = labels[i].item()
                score = scores[i].item()
                ax.text(xmin, ymin, f'{coco_names[label]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

        ax.axis('off')

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    plot_image_predictions(undefended_image, undefended_predictions, axes[0], "Undefended Model")
    plot_image_predictions(defended_image, defended_predictions, axes[1], "Defended Model")
    plt.show()

def erode(mask, kernel_size=11):
    # Apply erosion (min pooling)
    eroded_mask = F.max_pool2d(1 - mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
    return 1 - eroded_mask

def dilate(mask, kernel_size=11):
    # Apply dilation (max pooling)
    dilated_mask = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
    return dilated_mask

def apply_mean_kernel(mask, kernel_size=11):
    # Create a mean kernel
    kernel = torch.ones((1,1, kernel_size, kernel_size), device=mask.device) / (kernel_size * kernel_size)

    # Apply the mean kernel using convolution
    mean_filtered_mask = F.conv2d(mask.unsqueeze(0), kernel,padding=kernel_size // 2)
    return mean_filtered_mask.squeeze(0)

def AntiStyler_defense(image):

    # The AntiStyle Phase ( padding + antisyle model)
    image_height = image.shape[2]
    image_width = image.shape[3]
    original_size = (image_height, image_width)
    if IMSIZE is None:
        if PADDING != 0:
            content_image = torch.zeros((1, 3, image_height + 2 * PADDING, image_width + 2 * PADDING),device=get_device())
            content_image[:, :, :PADDING, :] = torch.rand((1, 3, PADDING, image_width+2*PADDING))
            content_image[:, :, -PADDING:, :] = torch.rand((1, 3, PADDING, image_width+2*PADDING))
            content_image[:, :, :, :PADDING] = torch.rand((1, 3, image_height+2*PADDING, PADDING))
            content_image[:, :, :, -PADDING:] = torch.rand((1, 3, image_height+2*PADDING, PADDING))
            content_image[:, :, PADDING:-PADDING, PADDING:-PADDING] = image
        else:
            content_image = image
    else:
        content_image = torch.zeros((1, 3, IMSIZE[0], IMSIZE[1]), device=get_device())
        if PADDING != 0:
            content_image[:, :, :PADDING, :] = torch.rand((1, 3, PADDING, IMSIZE[1]))
            content_image[:, :, -PADDING:, :] = torch.rand((1, 3, PADDING, IMSIZE[1]))
            content_image[:, :, :, :PADDING] = torch.rand((1, 3, IMSIZE[0], PADDING))
            content_image[:, :, :, -PADDING:] = torch.rand((1, 3, IMSIZE[0], PADDING))
            content_image[:, :, PADDING:-PADDING, PADDING:-PADDING] = resize(image, (IMSIZE[0] - 2 * PADDING, IMSIZE[1] - 2 * PADDING))
        else:
            content_image = resize(image, IMSIZE)

    content_image.to(get_device(), torch.float)
    style_image = torch.randn_like(content_image, device=get_device(), dtype=torch.float)
    anti_styled_image = antistyle_model.apply(content_image, style_image, content_image.clone())

    # The Filter Phase
    difference_image = torch.abs(anti_styled_image - content_image)
    difference_image = difference_image.mean(dim=1)
    flattened_values = difference_image.flatten()
    k = int(TOP_PERCENTILE * flattened_values.shape[0])
    threshold_value = torch.kthvalue(flattened_values, k, dim=0).values.item()
    difference_image = (difference_image >= threshold_value).float()
    difference_image = difference_image[:, PADDING:-PADDING, PADDING:-PADDING]

    # The Enhancement Phase
    difference_image = dilate(difference_image, kernel_size=11)
    difference_image = erode(difference_image, kernel_size=11)
    difference_image = apply_mean_kernel(difference_image, kernel_size=51)
    difference_image = (difference_image >= 0.5).float()
    difference_image = dilate(difference_image, kernel_size=11)

    # The Mask Phase
    percentage_above_threshold = torch.sum(difference_image).item()
    if percentage_above_threshold>0:
        difference_image = torch.stack([difference_image] * 3, dim=1)
        if PADDING != 0:
            anti_styled_image = anti_styled_image[:, :, PADDING:-PADDING, PADDING:-PADDING]
        final_image = anti_styled_image - (difference_image * anti_styled_image)
        if IMSIZE is not None:
            final_image = resize(final_image, original_size)
    else:
        final_image = image

    return final_image


if __name__ == "__main__":
    set_all_seed(123)
    torch.cuda.empty_cache()
    device = get_device()

    # Define Object Detection model
    faster_rcnn=fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    detection_model = faster_rcnn.eval().to(device)

    clean_image_path = "images/person/benign.jpg"
    adversarial_image_path = "images/person/adversarial.npy"

    '''Compare benign image processing'''
    # Load image
    image = load_image(clean_image_path)

    # Predict objects using non-defended model
    with torch.no_grad():
        undefended_prediction = detection_model(image.clone())

    # Apply defense
    defended_image = AntiStyler_defense(image.clone())
    with torch.no_grad():
        defended_prediction = detection_model(defended_image)

    plot_predictions(image, undefended_prediction,defended_image, defended_prediction)

    '''Compare adversarial image processing'''
    # Load image
    image = load_image(adversarial_image_path)

    # Predict objects using non-defended model
    with torch.no_grad():
        undefended_prediction = detection_model(image.clone())

    # Apply defense
    defended_image = AntiStyler_defense(image.clone())
    with torch.no_grad():
        defended_prediction = detection_model(defended_image)

    plot_predictions(image, undefended_prediction,defended_image, defended_prediction)


