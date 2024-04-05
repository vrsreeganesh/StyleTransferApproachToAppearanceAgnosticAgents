import argparse
import json
import os
from tqdm import tqdm
import time
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

#############################################################################################################################
import torch; import numpy as np; import matplotlib.pyplot as plt; import torchvision; 
import os; # to get path and stuff
import random; import datetime; # helps us name images
import PIL; # used in the image loader function

# Choosing device
if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
# ------------------------------------------------------------------------------------------------------------------------
# torch.set_default_device(device)
imsize0 = 160
imsize1 = 704
loader = torchvision.transforms.Compose([ torchvision.transforms.Resize((imsize0, imsize1)), torchvision.transforms.ToTensor()])
def image_loader(image_name):
    image = PIL.Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
class ContentLoss(torch.nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
    def forward(self, input):
        self.loss = torch.nn.functional.mse_loss(input, self.target)
        return input
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__(); self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input); self.loss = torch.nn.functional.mse_loss(G, self.target)
        return input
class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)

        # !!! This is my addition due to some compiler errors
        self.mean = mean.clone().detach().view(-1,1,1); self.std = std.clone().detach().view(-1,1,1)
    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# content_layers_default = ['conv_6']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               styleimage, contentimage,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []; style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = torch.nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            i += 1; name = 'conv_{}'.format(i)
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d): name = 'pool_{}'.format(i)
        elif isinstance(layer, torch.nn.BatchNorm2d): name = 'bn_{}'.format(i)
        else: raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(contentimage).detach(); content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss); content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(styleimage).detach(); style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss); style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       contentimage, styleimage, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    # Choosing device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # moving to gpu (just in case they already aren't)
    cnn = cnn.to(device); normalization_mean = normalization_mean.to(device); normalization_std = normalization_std.to(device);
    contentimage = contentimage.to(device); styleimage = styleimage.to(device); input_img = input_img.to(device);

    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, 
                                                                     normalization_std, 
                                                                     styleimage, 
                                                                     contentimage)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers 
    # such as dropout or batch normalization layers behave correctly. 
    model.eval(); model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    # print('Optimizing..')
    print("StyleLoss", end=  ": "); run = [0]
    num_steps = 250
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad(); model(input_img); style_score = 0; content_score = 0;
            for sl in style_losses: style_score += sl.loss; 
            for cl in content_losses: content_score += cl.loss
            style_score *= style_weight; content_score *= content_weight
            loss = style_score + content_score; loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                # print("run {}:".format(run))
                # print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                #     style_score.item(), content_score.item()))
                print('{:4f}'.format(style_score.item()), end = " | ")

            return style_score + content_score
        optimizer.step(closure)
    print()

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # optimizer = optim.LBFGS([input_img])
    optimizer = torch.optim.LBFGS([input_img], lr = 1e-1)
    return optimizer
def NSTSetup(styleimage, contentimage, content_weight_arg = 1e-2):
    # Choosing device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # moving things to GPU
    styleimage = styleimage.to(device); contentimage = contentimage.to(device);

    # cnn = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features.eval()
    cnn = torchvision.models.vgg19(pretrained=True).features.eval()
    cnn = cnn.to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]); cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    input_img = contentimage.clone()
    # plt.figure(); imshow(input_img, title='Input Image')

    # print("> type(contentimage) = ", type(contentimage))
    # print("> type(styleimage) = ", type(styleimage))
    # print("> styleimage.shape = ", styleimage.shape)
    # print("> contentimage.shape = ", contentimage.shape)
    # print("> torch.max(styleimage) = ", torch.max(styleimage))
    # print("> torch.max(contentimage) = ", torch.max(contentimage))

    # contentimage = contentimage.reshape([1,3,imsize0,imsize1])
    # styleimage = styleimage.reshape([1,3,imsize0,imsize1])
    # input_img = input_img.reshape([1,3,imsize0,imsize1])

    contentimage = contentimage.reshape([contentimage.shape[0],3,imsize0,imsize1])
    styleimage = styleimage.reshape([contentimage.shape[0],3,imsize0,imsize1])
    input_img = input_img.reshape([contentimage.shape[0],3,imsize0,imsize1])

    # output = run_style_transfer(cnn, 
    #                             cnn_normalization_mean,
    #                             cnn_normalization_std,
    #                             contentimage, 
    #                             styleimage, 
    #                             input_img,
    #                             style_weight=1e6,
    #                             content_weight=1e-2) #1e3, 1e2, 5e2, 1e3, 3e2, 2.5e2 is good, originally 1
    output = run_style_transfer(cnn, 
                                cnn_normalization_mean,
                                cnn_normalization_std,
                                contentimage, 
                                styleimage, 
                                input_img,
                                style_weight=1e6,
                                content_weight=content_weight_arg) #1e3, 1e2, 5e2, 1e3, 3e2, 2.5e2 is good, originally 1
    
    return output
def fetchcurrenttime():
    now = datetime.datetime.now(); formatted_date_time = now.strftime("%m_%d_%H_%M_%S");
    return formatted_date_time
#############################################################################################################################










from diskcache import Cache
# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=41, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--root_dir', type=str, default=r'/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11', help='Root directory of your training data')
    parser.add_argument('--schedule', type=int, default=1,
                        help='Whether to train with a learning rate schedule. 1 = True')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
    parser.add_argument('--backbone', type=str, default='transFuser',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
    parser.add_argument('--image_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--use_velocity', type=int, default=0,
                        help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--wp_only', type=int, default=0,
                        help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
    parser.add_argument('--parallel_training', type=int, default=1,
                        help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                             ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.')
    parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train normally, 1: set training weight for BEV to 0')
    parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
    parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')


    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    parallel = bool(args.parallel_training)

    if(bool(args.use_disk_cache) == True):
        if (parallel == True):
            # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
            # During training we cache the dataset on the fast storage of the local compute nodes.
            # Adapt to your cluster setup as needed.
            # Important initialize the parallel threads from torch run to the same folder (so they can share the cache).
            tmp_folder = str(os.environ.get('SCRATCH'))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            # We use a local diskcache to cache the dataset on the faster SSD drives on our cluster.
            shared_dict = Cache(directory=tmp_folder ,size_limit=int(768 * 1024 ** 3))
        else:
            shared_dict = Cache(size_limit=int(768 * 1024 ** 3))
    else:
        shared_dict = None
    # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
    if(parallel == True): #Non distributed works better with my local debugger
        rank       = int(os.environ["RANK"]) #Rank accross all processes
        local_rank = int(os.environ["LOCAL_RANK"]) # Rank on Node
        world_size = int(os.environ['WORLD_SIZE']) # Number of processes
        print(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")

        device = torch.device('cuda:{}'.format(local_rank))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank) # Hide devices that are not used by this process

        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,
                                             timeout=datetime.timedelta(minutes=15))

        torch.distributed.barrier(device_ids=[local_rank])
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:{}'.format(local_rank))
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True # Wen want the highest performance
    # Configure config
    '''
        args.root_dir = Root directory of your training data
        args.setting  = what kind of data to use: two options mainly. 
                        1. whether to keep a validation dataset
                        2. to keep two towns as validation set. 
    '''
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting) 
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone
    if(bool(args.no_bev_loss)):
        index_bev = config.detailed_losses.index("loss_bev")
        config.detailed_losses_weights[index_bev] = 0.0
    # Create model and optimizers
    model = LidarCenterNet(config, device, args.backbone, 
                           args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
    if (parallel == True):
        # Synchronizing the Batch Norms increases the Batch size with which they are compute by *num_gpus
        if(bool(args.sync_batch_norm) == True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    model.cuda(device=device)
    if ((bool(args.zero_redundancy_optimizer) == True) and (parallel == True)):

        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optim.AdamW, lr=args.lr) # Saves GPU memory during DDP training
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr) # For single GPU training
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)
    # Data
    train_set = CARLA_Data(root=config.train_data, config=config, shared_dict=shared_dict)
    val_set   = CARLA_Data(root=config.val_data,   config=config, shared_dict=shared_dict)
    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())
    if(parallel == True):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
        sampler_val   = torch.utils.data.distributed.DistributedSampler(val_set,   shuffle=True, num_replicas=world_size, rank=rank)
        dataloader_train = DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
        dataloader_val   = DataLoader(val_set,   sampler=sampler_val,   batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
    else:
      dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
      dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    # Create logdir
    if ((not os.path.isdir(args.logdir)) and (rank == 0)):
        print('Created dir:', args.logdir, rank)
        os.makedirs(args.logdir, exist_ok=True)

    # We only need one process to log the losses
    if(rank == 0):
        writer = SummaryWriter(log_dir=args.logdir)
        # Log args
        with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None

    if (not (args.load_file is None)):
        # Load checkpoint
        print("=============load=================")
        model.load_state_dict(torch.load(args.load_file, map_location=model.device))
        optimizer.load_state_dict(torch.load(args.load_file.replace("model_", "optimizer_"), map_location=model.device))


    trainer = Engine(model=model, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                     args=args, config=config, writer=writer, device=device, rank=rank, world_size=world_size,
                     parallel=parallel, cur_epoch=args.start_epoch)

    for epoch in range(trainer.cur_epoch, args.epochs):
        if(parallel == True):
            # Update the seed depending on the epoch so that the distributed sampler will use different shuffles across different epochs
            sampler_train.set_epoch(epoch)
        if ((epoch == args.schedule_reduce_epoch_01) or (epoch==args.schedule_reduce_epoch_02)) and (args.schedule == 1):
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1
            print("Reduce learning rate by factor 10 to:", new_lr)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        trainer.train()

        if((args.setting != 'all') and (epoch % args.val_every == 0)):
            trainer.validate()

        if (parallel == True):
            if (bool(args.zero_redundancy_optimizer) == True):
                optimizer.consolidate_state_dict(0) # To save the whole optimizer we need to gather it on GPU 0.
            if (rank == 0):
                trainer.save()
        else:
            trainer.save()

class Engine(object):
    """
    Engine that runs training.
    """

    def __init__(self, model, optimizer, dataloader_train, dataloader_val, args, config, writer, device, rank=0, world_size=1, parallel=False, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.parallel = parallel
        self.vis_save_path = self.args.logdir + r'/visualizations'
        if(self.config.debug == True):
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_losses         = config.detailed_losses
        if self.args.wp_only:
            detailed_losses_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            detailed_losses_weights = config.detailed_losses_weights
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

    def load_data_compute_loss(self, data):


        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # Move data to GPU
        rgb = data['rgb'].to(self.device, dtype=torch.float32)

        # making all the print statements be written to a file
        statusfilepath = ""
        statusfile = open(statusfilepath, 'a+') # opening the file
        sys.stdout = statusfile 
        print("time stamp = "+fetchcurrenttime() + "-------------------------    \n")
        
        # This section of code runs a certrain percentage of times. 
        # percentageoftimestostyletransfer = 0.7 # must be between 0 and 1
        percentageoftimestostyletransfer = 1 # must be between 0 and 1
        if random.random() < percentageoftimestostyletransfer:
            # getting the style images //////////////////////////////////////////////////////////////////////////////////////////////
            texturedirectory = ''
            stylefilenames = [texturedirectory+file for file in os.listdir(texturedirectory)]               # finding style file names
            stylerandomindices = [random.randint(0, len(stylefilenames)-1) for _ in range(rgb.shape[0])]    # choosing a random number of indices
            stylerandomnames = [stylefilenames[x] for x in stylerandomindices] # finding a random set of images
            for i in range(rgb.shape[0]):
                var00 = image_loader(stylerandomnames[i]); # loading an image
                print("torch.max(var00) = ", torch.max(var00))
                if i==0: styletensor = var00 # styletensor is the tensor we'll use to stack style tensors. 
                else: styletensor = torch.cat((styletensor, var00), axis = 0); # stacking style images to the styletensor
            
            # styletransferring ///////////////////////////////////////////////////////////////////////////////////////////////////
            '''This is where style transfer occurs.'''
            content_weight_arg = 1;
            output = NSTSetup(styletensor, rgb/255.0, content_weight_arg=content_weight_arg)
            print("torch.max(output) = ", torch.max(output))
            print("output.shape = ", output.shape)
            # colour mixing ///////////////////////////////////////////////////////////////////////////////////////////////////////
            '''
            Here, we're randomly permuting the channels so that the colours are randomly changed. Sure, this does cause issues in the traffic lights and other stuff, but I'm guessing it'll correlate the position of colours with the actions associated with green light and red light and whatever. 
            '''
            colourmixing = [var00.item() for var00 in torch.randperm(3)];
            print("colourmixing = ", colourmixing)
            output = output[:, colourmixing, :,:]
            # saving them /////////////////////////////////////////////////////////////////////////////////////////////////////////
            '''Here, we're saving them occassionally so that have an idea as to how these things look like. '''
            if random.random() < 0.05:
                # torchvision.utils.save_image(output, "/projectnb/rlvn/students/vrs/transfuser/RLVNFinalSubmission/STCarla/whole.png");
                gridimage = torchvision.utils.make_grid(output, nrow= 1, padding = 2); # changing the colours
                torchvision.utils.save_image(gridimage, ""+fetchcurrenttime()+".png");
            # since torch.max(output) = 1 and the rest of the code is expecting the image to be scaled to 255.0
            rgb = output*255.0
        sys.stdout = sys.__stdout__ # this basically returns the whole thing to original state
        statusfile.close()
        sys.exit("==================================== WOrk is done btw ====================================")
        # time.sleep(120);

        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        # ========================================================================================================================
        

        if self.config.multitask:
            depth = data['depth'].to(self.device, dtype=torch.float32)
            semantic = data['semantic'].squeeze(1).to(self.device, dtype=torch.long)
        else:
            depth = None
            semantic = None

        bev = data['bev'].to(self.device, dtype=torch.long)

        if (self.config.use_point_pillars == True):
            lidar = data['lidar_raw'].to(self.device, dtype=torch.float32)
            num_points = data['num_points'].to(self.device, dtype=torch.int32)
        else:
            lidar = data['lidar'].to(self.device, dtype=torch.float32)
            num_points = None

        label = data['label'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)

        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)

        ego_vel = data['speed'].to(self.device, dtype=torch.float32)

        if ((self.args.backbone == 'transFuser') or (self.args.backbone == 'late_fusion') or (self.args.backbone == 'latentTF')):
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                           target_point_image=target_point_image,
                           ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                           label=label, save_path=self.vis_save_path,
                           depth=depth, semantic=semantic, num_points=num_points)
        elif (self.args.backbone == 'geometric_fusion'):

            bev_points = data['bev_points'].long().to('cuda', dtype=torch.int64)
            cam_points = data['cam_points'].long().to('cuda', dtype=torch.int64)
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                           target_point_image=target_point_image,
                           ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                           label=label, save_path=self.vis_save_path,
                           depth=depth, semantic=semantic, num_points=num_points,
                           bev_points=bev_points, cam_points=cam_points)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        return losses


    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch  = {key: 0.0 for key in self.detailed_losses}
        self.cur_epoch += 1

        # Train loop
        for data in tqdm(self.dataloader_train):
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.load_data_compute_loss(data)
            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
            loss.backward()

            self.optimizer.step()
            num_batches += 1
            loss_epoch += float(loss.item())

        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')


    @torch.inference_mode() # Faster version of torch_no_grad
    def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch  = {key: 0.0 for key in self.detailed_losses}

        # Evaluation loop loop
        for data in tqdm(self.dataloader_val):
            losses = self.load_data_compute_loss(data)

            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_weights[key] * value.item())

            num_batches += 1
            loss_epoch += float(loss.item())

        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')
    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        # Average all the batches into one number
        loss_epoch = loss_epoch / num_batches
        for key, value in detailed_losses_epoch.items():
            detailed_losses_epoch[key] = value / num_batches

        # In parallel training aggregate all values onto the master node.

        gathered_detailed_losses = [None for _ in range(self.world_size)]
        gathered_loss = [None for _ in range(self.world_size)]

        if (self.parallel == True):
            torch.distributed.gather_object(obj=detailed_losses_epoch,
                                            object_gather_list=gathered_detailed_losses if self.rank == 0 else None, 
                                            dst=0)
            torch.distributed.gather_object(obj=loss_epoch, 
                                            object_gather_list=gathered_loss if self.rank == 0 else None,
                                            dst=0)
        else:
            gathered_detailed_losses[0] = detailed_losses_epoch
            gathered_loss[0] = loss_epoch
            
        if (self.rank == 0):
            # Log main loss
            aggregated_total_loss = sum(gathered_loss) / len(gathered_loss)
            self.writer.add_scalar(prefix + 'loss_total', aggregated_total_loss, self.cur_epoch)

            # Log detailed losses
            for key, value in detailed_losses_epoch.items():
                aggregated_value = 0.0
                for i in range(self.world_size):
                    aggregated_value += gathered_detailed_losses[i][key]

                aggregated_value = aggregated_value / self.world_size

                self.writer.add_scalar(prefix + key, aggregated_value, self.cur_epoch)
    def save(self):
        # NOTE saving the model with torch.save(model.module.state_dict(), PATH) if parallel processing is used would be cleaner, we keep it for backwards compatibility
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'model_%d.pth' % self.cur_epoch))
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.logdir, 'optimizer_%d.pth' % self.cur_epoch))

# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # The default method fork can run into deadlocks.
    # To use the dataloader with multiple workers forkserver or spawn should be used.
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
