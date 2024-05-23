# Third-party
import cv2
import numpy as np
import torch

# Standard
from pathlib import Path

# Project
import config as cf


async def detect_and_draw_boxes(image_path: Path, scale_factor: float = 1.0, name: str = 'blank.jpg') -> Path:
    """
    Detect objects in an image and draw bounding boxes with class names around detected objects.

    Parameters:
    - image_path: The path to the image file.
    - scale_factor: The factor by which the image size should be increased or decreased.
    - name: The name of the output image.

    Returns:
    - The original image with bounding boxes and class names drawn.
    - The list of detected labels.
    """
    # Initialize model variables
    class_names, colors, model = __init_model_vars()

    # Read image and resize
    image = cv2.imread(filename=str(image_path))
    image = __resize_image(image, scale_factor=scale_factor)
    image_height, image_width, _ = image.shape

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    model.setInput(blob)
    output = model.forward()

    # Initialize an empty list to hold detected labels
    detected_labels = []

    # Loop through the detections
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .4:
            class_id = detection[1]
            class_name = class_names[int(class_id) - 1]
            detected_labels.append(class_name)
            color = colors[int(class_id)]
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width - box_x
            box_height = detection[6] * image_height - box_y
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_x + box_width), int(box_y + box_height)), color,
                          thickness=2)
            cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Return the annotated image and detected labels
    result_path = cf.DATA_PATH / 'images' / name
    cv2.imwrite(filename=str(result_path), img=image)
    return result_path, detected_labels


def __init_model_vars() -> tuple[list[str], np.ndarray, cv2.dnn.Net]:
    """
    Initialize the model variables.

    Returns:
    - A tuple containing the class names, colors for each class, and the cv2 deep neural network model.
    """
    with open(str(cf.DATA_PATH / 'model' / 'object_detection_classes_coco.txt'), 'r') as f:
        class_names = f.read().split('\n')

    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    model = cv2.dnn.readNet(
        model=str(cf.DATA_PATH / 'model' / 'frozen_inference_graph.pb'),
        config=str(cf.DATA_PATH / 'model' / 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'),
        framework='TensorFlow'
    )

    return class_names, colors, model


def __resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Resize an image by a scale factor while maintaining the aspect ratio.

    Parameters:
    - image: The original image as a numpy array.
    - scale_factor: The factor by which the image size should be increased or decreased.

    Returns:
    - The resized image as a numpy array.
    """
    original_height, original_width, _ = image.shape
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    if original_height > original_width:
        new_width = int((new_height / original_height) * original_width)
    else:
        new_height = int((new_width / original_width) * original_height)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def test(loaders: dict, model: cv2.dnn.Net, criterion, use_cuda: bool) -> tuple[float, float]:
    """
    Test the trained model on validation and test data.

    Parameters:
    - loaders: Dictionary containing DataLoader objects with training, validation, and test data.
    - model: The trained model to be tested.
    - criterion: Loss function used to evaluate the model.
    - use_cuda: Boolean flag to indicate if GPU should be used for computations.

    Returns:
    - A tuple with loss and accuracy of the model.
    """
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data in loaders['test']:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples * 100

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy