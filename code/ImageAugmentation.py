import Augmentor
import datetime
import os
import cv2 as cv

class ImageAugmentor:
    def __init__(self):
        pass

    def __call__(self, image, unlabeled_path, new_label, samples=10):
        output = os.path.join(unlabeled_path, str(new_label))
        if not os.path.exists(output):
            os.mkdir(output)

        image_name = datetime.datetime.now().strftime("%H%M%S%f")
        cv.imwrite(os.path.join(output, f"{image_name}.jpg"), image)
        augmentor = Augmentor.Pipeline(output)
        augmentor.rotate(probability=0.9, max_left_rotation=25, max_right_rotation=25)
        augmentor.zoom(probability=0.9, min_factor=0.5, max_factor=2)
        augmentor.flip_left_right(probability=0.7)
        # augmentor.flip_top_bottom(probability=0.5)
        augmentor.shear(probability=0.8, max_shear_left=15, max_shear_right=15)
        augmentor.skew(probability=0.6, magnitude=0.4)
        augmentor.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=5)
        augmentor.gaussian_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=2, corner='bell', method='in')
        augmentor.random_brightness(probability=0.8, min_factor=0.7, max_factor=2)
        augmentor.random_contrast(probability=0.8, min_factor=0.7, max_factor=2)
        
        augmentor.sample(samples)

        images = []
        images.append(image)
        augmented_dir = os.path.join(output, 'output')
        files = os.listdir(augmented_dir)
        for file in files:
            images.append(cv.imread(os.path.join(augmented_dir, file)))
        
        return images