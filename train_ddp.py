import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.my_model import MyModel
from models.cut_model import CUTModel
from options.train_options import TrainOptions
from data.unaligned_dataset import UnalignedDataset



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
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
    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)

    opt = TrainOptions().parse()  # get training options
    
    # Custom model
    my_model = MyModel(opt).to(rank)
    ddp_model = DDP(my_model, device_ids=[rank], find_unused_parameters=True) if is_distributed else my_model

    # Create dataset
    dataset = UnalignedDataset(opt)  # create a dataset
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batch_size // world_size, 
        sampler=sampler,
        num_workers=int(opt.num_threads), 
        drop_last=True if opt.isTrain else False)
        # num_workers here refer to the number of CPU worker processes used per GPU/process to load data in parallel.
    if rank == 0:  # Only print dataset size from the main process
        print('The number of training images = %d' % len(dataloader.dataset))

    # Training loop
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            batch = move_dict_to_device(batch, rank)  # Move batch data to the current device (GPU)
            
            if epoch == opt.epoch_count and batch_idx == 0:
                ddp_model.module.data_dependent_initialize(**batch)
                ddp_model.module.setup(opt) # regular setup: load and print networks; create schedulers

            ddp_model.module.set_input(**batch)  # unpack data from dataset and apply preprocessing
            ddp_model.module.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if rank == 0:
            print(f"Epoch {epoch} done.")

    if is_distributed:
        cleanup()




# Entry point
def main():
    # rank = int(os.environ["RANK"])  # Get the rank of the current process
    # world_size = int(os.environ["WORLD_SIZE"])  # Get the total number of processes (or GPUs)
    # train_sample(rank, world_size)
    
    opt = TrainOptions().parse()  # get training options
    world_size = opt.nprocs_per_node * opt.nnodes  # world_size is the total number of processes (or GPUs) to use on all server nodes
    
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)

if __name__ == "__main__":
    main()