import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms


class AdversarialGenerator:
    def __init__(self, model=torchvision.models.resnet34(weights='IMAGENET1K_V1'), dataset_path="../data", device=None):
        self.model = model.to(self._get_device())
        self.dataset_path = dataset_path
        self.device = self._get_device()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.norm_mean = np.array([0.485, 0.456, 0.406])
        self.norm_std = np.array([0.229, 0.224, 0.225])
        self.plain_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.dataset = self._load_dataset()
        self.label_names = self._load_label_names()

    def _get_device(self):
        return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    def _download_files(self, files):
        base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
        for dir_name, file_name in files:
            file_path = os.path.join(dir_name, file_name)
            if not os.path.isfile(file_path):
                file_url = base_url + file_name
                print(f"Downloading {file_url}...")
                try:
                    urllib.request.urlretrieve(file_url, file_path)
                except HTTPError as e:
                    print("Something went wrong. Please try to download the file from the GDrive folder: \n", e)
                if file_name.endswith(".zip"):
                    print("Unzipping file...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(file_path.rsplit("/", 1)[0])

    def _load_dataset(self):
        imagenet_path = os.path.join(self.dataset_path, "TinyImageNet/")
        assert os.path.isdir(imagenet_path), f"Could not find the ImageNet dataset at expected path \"{imagenet_path}\". " + \
                                             f"Please make sure to have downloaded the ImageNet dataset here, or change the dataset_path variable."
        return torchvision.datasets.ImageFolder(root=imagenet_path, transform=self.plain_transforms)

    def _load_label_names(self):
        imagenet_path = os.path.join(self.dataset_path, "TinyImageNet/")
        with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
            return json.load(f)

    def get_label_index(self, lab_str):
        assert lab_str in self.label_names, f"Label \"{lab_str}\" not found. Check the spelling of the class."
        return self.label_names.index(lab_str)

    def preprocess_input(self, img):
        return img

    def clip_eps(self, delta, eps):
        return torch.clamp(delta, -eps, eps)

    def show_prediction(self, img, label, pred, K=5, adv_img=None, noise=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            img = (img * self.norm_std[None, None]) + self.norm_mean[None, None]
            img = np.clip(img, a_min=0.0, a_max=1.0)

        if noise is None or adv_img is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 1]})
        else:
            fig, ax = plt.subplots(1, 5, figsize=(12, 2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})

        ax[0].imshow(img)
        ax[0].set_title(self.label_names[label])
        ax[0].axis('off')

        if adv_img is not None and noise is not None:
            adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
            adv_img = (adv_img * self.norm_std[None, None]) + self.norm_mean[None, None]
            adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
            ax[1].imshow(adv_img)
            ax[1].set_title('Adversarial')
            ax[1].axis('off')
            noise = noise.cpu().permute(1, 2, 0).numpy()
            noise = noise * 0.5 + 0.5
            ax[2].imshow(noise)
            ax[2].set_title('Noise')
            ax[2].axis('off')
            ax[3].axis('off')

        if abs(pred.sum().item() - 1.0) > 1e-4:
            pred = torch.softmax(pred, dim=-1)
        topk_vals, topk_idx = pred.topk(K, dim=-1)
        topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
        ax[-1].barh(np.arange(K), topk_vals * 100.0, align='center', color=["C0" if topk_idx[i] != label else "C2" for i in range(K)])
        ax[-1].set_yticks(np.arange(K))
        ax[-1].set_yticklabels([self.label_names[c] for c in topk_idx])
        ax[-1].invert_yaxis()
        ax[-1].set_xlabel('Confidence')
        ax[-1].set_title('Predictions')

        plt.show()

    def generate_targeted_adversaries(self, base_image, delta, class_idx, target, steps=200, epsilon=0.05, learning_rate=0.01):
        base_image = base_image.to(self.device)
        delta = delta.to(self.device).requires_grad_()
        optimizer = optim.SGD([delta], lr=learning_rate)

        for step in range(steps + 1):
            optimizer.zero_grad()
            adversary = self.preprocess_input(base_image + delta)
            predictions = self.model(adversary.to(self.device))
            original_loss = -F.cross_entropy(predictions, torch.tensor([class_idx], device=self.device))
            target_loss = F.cross_entropy(predictions, torch.tensor([target], device=self.device))
            total_loss = original_loss + target_loss

            if step % 20 == 0:
                print(f"step: {step}, loss: {total_loss.item()}")

            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(self.clip_eps(delta, epsilon))

        adversarial_image = delta.detach()
        with torch.no_grad():
            self.show_prediction(base_image.squeeze(), class_idx, predictions[0], adv_img=adversary.squeeze(), noise=delta.squeeze())

        return adversarial_image

    def normal_prediction(self, base_image, class_idx):
        base_image = base_image.to(self.device)
        predictions = self.model(base_image)
        with torch.no_grad():
            self.show_prediction(base_image.squeeze(), class_idx, predictions[0])

        return

if __name__ == "__main__":
    adv_gen = AdversarialGenerator()
    parser = argparse.ArgumentParser(description="Generate targeted adversarial examples")
    parser.add_argument("--base_image", type=str, default=adv_gen.dataset[0][0].unsqueeze(0), help="Path to the base image")
    parser.add_argument("--delta", type=str, default=torch.zeros_like(adv_gen.dataset[0][0].unsqueeze(0)), help="Initial perturbation")
    parser.add_argument("--class_idx", type=int, default=adv_gen.dataset[0][1], help="Class index of the base image")
    parser.add_argument("--target", type=int, default=adv_gen.get_label_index("banana"), help="Target class label")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation limit")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for perturbation updates")
    args = parser.parse_args()    

    adv_gen.generate_targeted_adversaries(base_image=args.base_image, delta=args.delta, class_idx=args.class_idx, target=args.target, steps=args.steps, epsilon=args.epsilon, learning_rate=args.learning_rate)