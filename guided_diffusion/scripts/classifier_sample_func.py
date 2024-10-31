"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from ..guided_diffusion import dist_util, logger
from ..guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    #classifier_defaults,
    create_model_and_diffusion,
    #create_classifier,
    add_dict_to_argparser,
    #args_to_dict,
)
def args_to_dict(args, keys):
    return {k: args[k] for k in keys}

def classifier_sampler(epoch, class_sample_list, classifier_model):
    #args = create_argparser().parse_args()
    args = dict(
        clip_denoised=True,
        num_samples=2000,
        batch_size=32,
        use_ddim=True,
        model_path="/home/slidm/DDPM/guided-diffusion-miccai/checkpoints/isic2018_128dm_thesis/ema_0.9999_050000.pt",
        classifier_scale=1.0,
        class_num=7,
        save_img_dir='./results/128cs5w_onlinedm_0.2_ddim50',
        timestep_respacing="ddim50",
        image_size=128,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    #args.update(model_and_diffusion_defaults())
    
    args['num_samples'] = len(class_sample_list)
    #args['timestep_respacing'] = "ddim50"
    #args['learn_sigma'] = True
    #args['image_size'] = 128
    
    print(args['model_path'])
    print(args['image_size'])
    print(args['num_res_blocks'])
    print(args['attention_resolutions'])


    dist_util.setup_dist()
    logger.configure(dir=args['save_img_dir'])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args['model_path'], map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    if args['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    #classifier = create_classifier(**args_to_dict(args, #classifier_defaults().keys()))
    #classifier.load_state_dict(
    #    dist_util.load_state_dict(args.classifier_path, #map_location="cpu")
    #)
    #classifier.to(dist_util.dev())
    #if args.classifier_use_fp16:
    #    classifier.convert_to_fp16()
    #classifier.eval()



    def cond_fn(x, t, y=None):
        assert y is not None
        y = y.type(th.long)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier_model(x_in,return_logits=True)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args['classifier_scale']

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args['class_cond'] else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    classes_all = th.IntTensor(class_sample_list).to(dist_util.dev())
    batch = 0
    while len(all_images) * args['batch_size'] < args['num_samples']-args['batch_size']:
        model_kwargs = {}
        #classes = th.randint(
        #    low=0, high=args.class_num, size=(args.batch_size,), #device=dist_util.dev()
        #)
        #classes = th.IntTensor(class_sample_list)
        classes = classes_all[batch*args['batch_size']:(batch+1)*args['batch_size']]
        batch += 1
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args['batch_size'], 3, args['image_size'], args['image_size']),
            clip_denoised=args['clip_denoised'],
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args['batch_size']} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args['num_samples']]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args['num_samples']]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_epoch{str(epoch)}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")
    return out_path


#if __name__ == "__main__":
#    main()
