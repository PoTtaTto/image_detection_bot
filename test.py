# Project imports
from detection import detect_and_draw_boxes
from config import DATA_PATH

# Standard imports
import unittest
from json import loads
import time


TEST_PATH = DATA_PATH / 'images' / 'test'


class TestDetectionAccuracy(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def print_accuracy(accuracy, total_images, correct_detections, total_time):
        """
        Prints the accuracy of the detection along with the total time taken.

        Args:
        - accuracy (float): The accuracy percentage.
        - total_images (int): The total number of images tested.
        - correct_detections (int): The number of correct detections.
        - total_time (float): The total time taken for detection in seconds.
        """
        print(f'Точность: {accuracy / total_images:.2f}. Верно {correct_detections} из {total_images}. Потрачено {total_time} сек.')


    async def test_detection_accuracy_async(self):
        """
        Asynchronously tests the accuracy of the detect_and_draw_boxes function by comparing the detected labels
        with the ground truth labels for a set of image paths.

        Assertions:
        - The test verifies if the accuracy meets the expected value.

        """
        img_paths = list(TEST_PATH.glob('*.png'))

        with open(TEST_PATH / 'labels.json') as f:
            test_labels = loads(f.read())

        accuracy = 0
        start_time = time.time()
        
        # Loop through the image paths and verify the detected labels
        for path in img_paths:
            _, labels = await detect_and_draw_boxes(image_path=path)
            if labels == test_labels[path.name]:
                accuracy += 1
        
        end_time = time.time()
        total_time = round(end_time - start_time, 2)

        # # Print the accuracy and the time taken
        self.print_accuracy(accuracy / len(img_paths), len(img_paths), accuracy, total_time)

        # Here we could add assertions based on expected accuracy
        # For example, let's assert we have at least 90% accuracy
        expected_accuracy = 0.90
        self.assertGreaterEqual(accuracy / len(img_paths), expected_accuracy, f"Expected at least {expected_accuracy*100}% accuracy")


if __name__ == '__main__':
    unittest.main()
