import os
import torch
import torch.distributed as dist

def my_function(rank, world_size):
    # Your code here
    print(f"Running function on Rank {rank} out of {world_size} processes.")


def main():
    before = os.environ.get("LOCAL_RANK", None)
    print(f"LOCAL RANK before = {before}")


    dist.init_process_group('nccl')

    
    after = os.environ.get("LOCAL_RANK", None)
    print(f"LOCAL RANK after = {after}")
    
    # Get the current rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"rank = {rank}")
    print(f"world_size = {world_size}")

    # Call the function on all ranks
    my_function(rank, world_size)

if __name__ == '__main__':
    main()



