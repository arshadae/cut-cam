import os
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.unaligned_dataset import UnalignedDataset # Shit to be handled


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' # Only suitable for single node. replace with real addresses for multinode setup
    os.environ['MASTER_PORT'] = '12357'  # Use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def move_dict_to_device(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

def train(rank, world_size):
    torch.autograd.set_detect_anomaly(True)
    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)
    opt = TrainOptions().parse()   # get training options

    dataloader, sampler = create_dataset(opt, is_distributed, rank, world_size)
    dataset_size = len(dataloader.dataset)
    # model = create_model(opt)      # create a model given opt.model and other options
    # print('The number of training images = %d' % dataset_size)
    # Create model
    my_model = create_model(opt).to(rank)  # create a model given opt.model and other options
    first_batch = next(iter(dataloader))  # Get a batch of data
    first_batch = move_dict_to_device(first_batch, rank)  # Move batch data to the current device (GPU)
    my_model.data_dependent_initialize(first_batch)
    my_model.setup(opt) # regular setup: load and print networks; create schedulers
    
    if is_distributed:
        # Wrap the model with DistributedDataParallel
        # Note: find_unused_parameters=True is used if your model has branches that may not be used in every forward pass
        # This is useful for models with multiple outputs or conditional branches.
        # If your model does not have such branches, you can set find_unused_parameters=False for better performance.
        # If you are not sure, you can start with find_unused_parameters=True and then optimize later.
        my_model.netG = DDP(my_model.netG, device_ids=[rank], find_unused_parameters=False)
        if hasattr(my_model, 'netD'):
            my_model.netD = DDP(my_model.netD, device_ids=[rank], find_unused_parameters=False)
        if hasattr(my_model, 'netF'):
            my_model.netF = DDP(my_model.netF, device_ids=[rank], find_unused_parameters=False)
        if hasattr(my_model, 'netC'):
            my_model.netC = DDP(my_model.netC, device_ids=[rank], find_unused_parameters=False)  


    # ddp_model = DDP(my_model, device_ids=[rank], find_unused_parameters=True) if is_distributed else my_model
    # model = ddp_model.module if is_distributed else ddp_model
    model = my_model  # Use the model directly, as it is already wrapped in DDP if needed


    if rank == 0:  # Only print dataset size from the main process
        print('The number of training images = %d' % dataset_size)
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        # opt.visualizer = visualizer
    else:
        visualizer = None
 
    total_iters = 0                # the total number of training iterations
    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        if is_distributed:
            sampler.set_epoch(epoch)    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        
        if rank == 0:
            print('Training epoch %d / %d' % (epoch, opt.n_epochs + opt.n_epochs_decay))
            # print(opt.name)
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # dataset.set_epoch(epoch) # Check this shit
        for i, data in enumerate(dataloader):  # inner loop within one epoch
            
            if rank == 0:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                batch_size = data["A"].size(0) * world_size
                total_iters += batch_size
                epoch_iter += batch_size
                optimize_start_time = time.time()
            
            data = move_dict_to_device(data, rank)  # Move batch data to the current device (GPU)
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if rank == 0:
                optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                    if opt.display_id is None or opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)  # it's useful to occasionally show the experiment name on console
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

        if rank == 0:
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epo
    if is_distributed:
        cleanup()

# Entry point
def main():    
    opt = TrainOptions().parse()  # get training options
    world_size = opt.nprocs_per_node * opt.nnodes  # world_size is the total number of processes (or GPUs) to use on all server nodes
    
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)

if __name__ == "__main__":
    main()