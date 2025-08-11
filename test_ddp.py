import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def main_worker(rank, world_size, opt):
    is_distributed = world_size > 1
    setup_distributed(rank, world_size)

    # Adjust opt for multi-GPU
    opt.num_threads = 1  # can be > 0 for DDP
    opt.batch_size = 16
    opt.serial_batches = False  # sampler handles shuffling
    opt.no_flip = True
    opt.display_id = -1

    # Dataset with DistributedSampler
    # dataset = create_dataset(opt)  # this must support sampler injection
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, num_workers=opt.num_threads)

    dataloader, sampler = create_dataset(opt, is_distributed, rank, world_size)

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Each rank saves to its own directory
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f'{opt.phase}_{opt.epoch}_rank{rank}')
    if rank == 0:
        print(f'creating web directory {web_dir}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if rank == 0 and i % 5 == 0:
            print(f'[Rank {rank}] processing ({i:04d}) {img_path}')

        save_images(webpage, visuals, img_path, width=opt.display_winsize)

    webpage.save()
    cleanup_distributed()

if __name__ == '__main__':
    opt = TestOptions().parse()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker,
                                args=(world_size, opt),
                                nprocs=world_size,
                                join=True)
