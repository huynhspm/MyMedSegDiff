

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import imageio
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size), antialias=True),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test,test_flag=True)
        args.in_ch = 5
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")
    logger.log(args.model_path)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    
    # paths = ['0000056', '0009994', '0010018', '0010255', '0010038', 
    #          '0000101', '0010591', '0010584', '0010058', '0011374', 
    #          '0008116', '0010597', '0009993', '0009902', '0011143', 
    #          '0011175', '0008998', '0011132', '0000487', '0000490', 
    #          '0010347', '0002673', '0001242', '0000515', '0000378', 
    #          '0000534', '0000136', '0010192', '0001484', '0010261', 
    #          '0000125', '0011333', '0010202', '0000418', '0010073', 
    #          '0011151', '0009958', '0000188', '0011112', '0011310', 
    #          '0000172', '0010846', '0003559', '0009990', '0000226', 
    #          '0000501', '0000320', '0000003', '0011325', '0008626']
    variances = []
    
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data"
        
        if args.data_name == 'ISIC':
            # only 50 samples
            # if not (path[0][-11:-4] in paths):
            #     continue
        
        print(b.shape, m.shape)
        
        o1, o2, o3, o4 = b.split(4, dim=1)
        print(o1.shape, o2.shape)
        return
        
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")
        logger.log(path[0])
    
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        if args.debug:
            img_folder = os.path.join(args.out_dir, str(slice_ID))
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            
        times = []
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out, scores, samples = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            
            time_consume = start.elapsed_time(end)
            times.append(time_consume / 60000)
            print('time for 1 sample', time_consume / 60000)  #time measurement for the generation of 1 sample

        
            co = cal_out.clone().detach() 
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
            else:
                enslist.append(co)

            if args.debug:
                if args.data_name == 'ISIC':
                    o = org.clone().detach()[:,:-1,:,:]
                    c = cal.clone().detach().repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    n_batch,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(n_batch, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    mm = m.to(o.device).repeat(1, 3, 1, 1)
                    tup = (mm, o, ss, c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(img_folder, f"{i}.jpg"), nrow = 4, padding = 10)
                
                ids = th.linspace(0, len(scores) - 1, 20).to(th.int64)
                scores = th.cat(scores, dim=0).moveaxis(1, 3).cpu().squeeze()
                samples = th.cat(samples, dim=0).moveaxis(1, 3).cpu().squeeze()
                scores = scores[ids]
                samples = samples[ids]
                
                imageio.mimsave(f"{img_folder}/scores_{i}.gif", list(scores))
                imageio.mimsave(f"{img_folder}/samples_{i}.gif", list(samples))
                
        variance = th.cat(enslist,dim=0).var(dim=0, keepdim=True)
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        
        if args.debug:
            if args.data_name == 'ISIC':
                compose = th.cat((m.to(ensres.device).repeat(1, 3, 1, 1), 
                                  b.to(ensres.device), 
                                  ensres.repeat(1, 3, 1, 1), 
                                  variance.repeat(1, 3, 1, 1)), dim = 0)
                vutils.save_image(compose, fp = os.path.join(img_folder, "ensemble.jpg"), nrow = 4, padding = 10)

        # for evaluation
        output_folder = os.path.join(args.out_dir, 'eval')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        vutils.save_image(ensres, fp = os.path.join(output_folder, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)

        logger.log(times)
        avg_var = variance.mean().cpu().item()
        logger.log(avg_var)
        variances.append(avg_var)
    
    average_variance = sum(variances) / len(variances)
    logger.log(average_variance)
        
def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
