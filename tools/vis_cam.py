import os
import sys
sys.path.insert(0,os.getcwd())
import argparse
import copy
import math
import pkg_resources
import re
from pathlib import Path
import torch
# from PIL import Image
# from torchvision import transforms
import cv2
import numpy as np

from models.build import BuildNet
from utils.version_utils import digit_version
from utils.train_utils import  file2dict
from utils.misc import to_2tuple
from utils.inference import init_model
from core.datasets.compose import Compose
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm


try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the norm layer in the last block. Backbones '
        'implemented by users are recommended to manually specify'
        ' target layers in commmad statement.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=[],
        nargs='+',
        type=int,
        help='The target category to get CAM, default to use result '
        'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--vit-like',
        action='store_true',
        help='Whether the network is a ViT-like network.')
    parser.add_argument(
        '--num-extra-tokens',
        type=int,
        help='The number of extra tokens in ViT-like backbones. Defaults to'
        ' use num_extra_tokens of the backbone.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def build_reshape_transform(model, args):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    # ViT_based_Transformers have an additional clstoken in features
    if not args.vit_like:

        def check_shape(tensor):
            assert len(tensor.size()) != 3, \
                (f"The input feature's shape is {tensor.size()}, and it seems "
                 'to have been flattened or from a vit-like network. '
                 "Please use `--vit-like` if it's from a vit-like network.")
            return tensor

        return check_shape

    if args.num_extra_tokens is not None:
        num_extra_tokens = args.num_extra_tokens
    elif hasattr(model.backbone, 'num_extra_tokens'):
        num_extra_tokens = model.backbone.num_extra_tokens
    else:
        num_extra_tokens = 1

    def _reshape_transform(tensor):
        """reshape_transform helper."""
        assert len(tensor.size()) == 3, \
            (f"The input feature's shape is {tensor.size()}, "
             'and the feature seems not from a vit-like network?')
        tensor = tensor[:, num_extra_tokens:, :]
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        assert height * height == heat_map_area, \
            (f"The input feature's length ({heat_map_area+num_extra_tokens}) "
             f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
             ' which is not a perfect square number. Please check if you used '
             'a wrong num-extra-tokens.')
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return _reshape_transform

def apply_transforms(img_path, pipeline_cfg):
    """Apply transforms pipeline and get both formatted data and the image
    without formatting."""
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        """to split the transfoms into image_transforms and
        format_transforms."""
        image_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMS_SET:
                format_transforms_cfg.append(transform)
            else:
                image_transforms_cfg.append(transform)
        return image_transforms_cfg, format_transforms_cfg

    image_transforms, format_transforms = split_pipeline_cfg(pipeline_cfg)
    image_transforms = Compose(image_transforms)
    format_transforms = Compose(format_transforms)

    intermediate_data = image_transforms(data)
    inference_img = copy.deepcopy(intermediate_data['img'])
    format_data = format_transforms(intermediate_data)

    return format_data, inference_img

class MMActivationsAndGradients(ActivationsAndGradients):
    """Activations and gradients manager for mmcls models."""

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(
            x, return_loss=False, softmax=False, post_process=False)


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object."""

    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Release the original hooks in ActivationsAndGradients to use
    # MMActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = MMActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    cur_layer = model
    layer_names = layer_str.strip().split('.')

    def get_children_by_name(model, name):
        try:
            return getattr(model, name)
        except AttributeError as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    def get_children_by_eval(model, name):
        try:
            return eval(f'model{name}', {}, {'model': model})
        except (AttributeError, IndexError) as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    for layer_name in layer_names:
        match_res = re.match('(?P<name>.+?)(?P<indices>(\\[.+\\])+)',
                             layer_name)
        if match_res:
            layer_name = match_res.groupdict()['name']
            indices = match_res.groupdict()['indices']
            cur_layer = get_children_by_name(cur_layer, layer_name)
            cur_layer = get_children_by_eval(cur_layer, indices)
        else:
            cur_layer = get_children_by_name(cur_layer, layer_name)

    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    
    grayscale_cam = grayscale_cam[0, :]

    src_img = np.float32(src_img)[:, :, ::-1] / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        cv2.imwrite(str(out_path), visualization_img)
    else:
        cv2.imshow(title, visualization_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_default_traget_layers(model, args):
    """get default target layers from given model, here choose norm type layer
    as default target layer."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')
    # if the model is CNN model or Swin model, just use the last norm
    # layer as the target-layer, if the model is ViT model, the final
    # classification is done on the class token computed in the last
    # attention block, the output will not be affected by the 14x14
    # channels in the last layer. The gradient of the output with
    # respect to them, will be 0! here use the last 3rd norm layer.
    # means the first norm of the last decoder block.
    if args.vit_like:
        if args.num_extra_tokens:
            num_extra_tokens = args.num_extra_tokens
        elif hasattr(model.backbone, 'num_extra_tokens'):
            num_extra_tokens = model.backbone.num_extra_tokens
        else:
            raise AttributeError('Please set num_extra_tokens in backbone'
                                 " or using 'num-extra-tokens'")

        # if a vit-like backbone's num_extra_tokens bigger than 0, view it
        # as a VisionTransformer backbone, eg. DeiT, T2T-ViT.
        if num_extra_tokens >= 1:
            print('Automatically choose the last norm layer before the '
                  'final attention block as target_layer..')
            return [norm_layers[-3]]
    print('Automatically choose the last norm layer as target_layer.')
    target_layers = [norm_layers[-1]]
    return target_layers


def main():
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    # apply transform and perpare data
    data, src_img = apply_transforms(args.img, val_pipeline)

    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_traget_layers(model, args)

    # init a cam grad calculator
    use_cuda = ('cuda' in args.device)
    reshape_transform = build_reshape_transform(model, args)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform)

    # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7,
    # to fix the bug in #654.
    targets = None
    if args.target_category:
        grad_cam_v = pkg_resources.get_distribution('grad_cam').version
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(c) for c in args.target_category]
        else:
            targets = args.target_category

    # calculate cam grads and show|save the visualization image
    grayscale_cam = cam(
        data['img'].unsqueeze(0),
        targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)
    show_cam_grad(
        grayscale_cam, src_img, title=args.method, out_path=args.save_path)


if __name__ == '__main__':
    main()

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets_3M/test/DR/10454.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10454.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/AMD/10456.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10456.bmp --target-category 3

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/AMD/10458.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10458.bmp --target-category 3

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/DR/10464.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10464.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/DR/10467.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10467.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/CNV/10473.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10473.bmp --target-category 1

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/CNV/10474.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10474.bmp --target-category 1

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/DR/10479.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10479.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/DR/10489.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10489.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/DR/10499.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10499.bmp --target-category 2

# python tools/vis_cam.py /home/slidm/OCTA/Awesome-Backbones/datasets/test/NORMAL/10483.bmp /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet18_b53M.py --save-path /home/slidm/OCTA/Awesome-Backbones/cam_results/10483.bmp --target-category 0