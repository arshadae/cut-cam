import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from models.sample_models import MyModel
from options.train_options import TrainOptions
from data.unaligned_dataset import UnalignedDataset
from models.custom_loss_criterion import HardThresholdLoss  # Custom loss function


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # Use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # "nccl" is GPU only backend, "gloo" is CPU and GPU compatible
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def train(rank, world_size):
    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)

    opt = TrainOptions().parse()  # get training options
    
    # Custom model
    my_model = MyModel().to(rank)
    ddp_model = DDP(my_model, device_ids=[rank])

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


    criterion = HardThresholdLoss() # Loss functions in PyTorch are device agnostic
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001) # Optimizers in PyTorch are also device agnostic
    
    # Training loop
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            AtoB = opt.direction == 'AtoB'
            inputs, targets, _, _ = batch['A' if AtoB else 'B'], batch['B' if AtoB else 'A'], batch['A_paths'], batch['B_paths']
            inputs, targets = inputs.to(rank), targets.to(rank)

            outputs = ddp_model(inputs)            
            # Forward pass: it internally calls ddp_model.forward(inputs)
            if rank == 0 and epoch ==0:  # Only print shapes from the main process
                print(f"input shape: {inputs.shape}, output shape: {outputs.shape}, target shape: {targets.shape}")


            loss_tensor = criterion(outputs, targets) 
            # Calling forward method of the loss class, returns a Tensor with requires_grad=True
            # loss_tensor is a scalar tensor with a gradient function attached to it
            # print(f"Loss grad_fn: {loss_tensor.grad_fn}") # <NllLossBackward> or similar

            optimizer.zero_grad() # Clear previous gradients       
            loss_tensor.backward() # Backpropagation, traiggers autograd to compute new gradients
            # print(f"outputs grad_fn: {outputs.grad_fn}") 
            # Now outputs also has gradients attached to it
            
            optimizer.step() # Update model parameters (weights) based on the computed gradients

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