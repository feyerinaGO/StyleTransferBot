from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import copy
import asyncio
import random
from nmodel import Normalization, ContentLoss, StyleLoss


class StyleModel:
    def __init__(self, content_path, style_path, model_path='nmodel/model/vgg19.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.content_path = content_path
        self.style_path = style_path
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnt_loader = transforms.Compose([transforms.ToTensor()])
        self.unloader = transforms.ToPILImage()

    async def content_loader(self, image_name):
        image = Image.open(image_name)
        image = self.cnt_loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    async def style_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    async def load_images(self):
        self.content_img = await self.content_loader(self.content_path)
        self.img_size_x = self.content_img.size()[2]
        self.img_size_y = self.content_img.size()[3]
        self.loader = transforms.Compose([
            transforms.Resize([self.img_size_x, self.img_size_y]),
            transforms.CenterCrop([self.img_size_x, self.img_size_y]),
            transforms.ToTensor()])
        self.style_img = await self.style_loader(self.style_path)
        self.input_img = self.content_img.clone()

    async def get_style_model_and_losses(self):
        content_layers = self.content_layers_default
        style_layers = self.style_layers_default
        cnn = copy.deepcopy(self.model)


        normalization = Normalization.Normalization(self.normalization_mean, self.normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
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

            if name in content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss.ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss.StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss.ContentLoss) or isinstance(model[i], StyleLoss.StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    async def get_input_optimizer(self):
        self.optimizer = optim.LBFGS([self.input_img.requires_grad_()])

    async def run_style_transfer(self, num_steps=300,
                           style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = await self.get_style_model_and_losses()
        await self.get_input_optimizer()

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            await asyncio.sleep(random.randint(0, 5))

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                self.input_img.data.clamp_(0, 1)

                self.optimizer.zero_grad()

                model(self.input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print(f'Style Loss : {style_score} Content Loss: {content_score}')
                    print()

                return style_score + content_score

            self.optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)

    async def image_output(self):
        image = self.input_img.clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        return image


