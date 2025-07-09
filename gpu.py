

import pynvml
import torch





def get_free_memory(device):
    """
    Given a device name (e.g. 'cuda:0'), returns how many GB of free memory are available on that device.
    """
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device[-1]))
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_GB = info.free/1024**3
    pynvml.nvmlShutdown()
    
    return free_GB


def choose_gpu():

    """
    Finds the GPU with the most free memory, and returns the corresponding device name (e.g. 'cuda:0').
    If no GPU is available, returns 'cpu'.
    """
    
    if not torch.cuda.is_available():
        print(f'No GPU available, using CPU')
        return 'cpu'
    


    pynvml.nvmlInit()

    ngpus = pynvml.nvmlDeviceGetCount()
    cuda_id_best = 0
    mem_best = 0.0

    for cuda_id in range(ngpus):

        handle = pynvml.numlDeviceGetHandleByIndex(cuda_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f'GPU {cuda_id}: {info.free/1024**3:.2f} GB free')

        if info.free > mem_best:
            cuda_id_best = cuda_id
            mem_best = info.free

    pynvml.nvmlShutdown()
    print(f'Best GPU: {cuda_id_best}')

    return f'cuda:{cuda_id_best}' if torch.cuda.is_available() else 'cpu'





