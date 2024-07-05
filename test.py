from adversarial_attack_library import AdversarialGenerator
import torchvision
import torch

"""
The following script shows how to test the AdversarialGenerator class 
by requesting a sample image (first image of the TinyImageNet dataset)
"""

# load the dataset from AdversarialGenerator class
dataset = AdversarialGenerator().dataset
# load the first image of the TinyImagenet dataset as our example
base_image = dataset[0][0].unsqueeze(0)
# load the delta vector (initially all zeros) which is our vector to add noise for the sake of our adversarial attack.
delta = torch.zeros_like(base_image)
# load the actual label of the sample image
class_idx = dataset[0][1]
# what is the target in which we want the sample image to be classified as
target = AdversarialGenerator().get_label_index('banana')

# The following command calls the generate_targeted_adversaries, this is where the delta vector gets trained to add noise to the original image
AdversarialGenerator().generate_targeted_adversaries(base_image, delta, class_idx, target, steps = 200)

# The following command calls normal_prediction to see what is the actual prediction of the image without any noise
AdversarialGenerator().normal_prediction(base_image, class_idx)