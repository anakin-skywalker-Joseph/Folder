import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm
import ipdb


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,           ## epoch
    throw_out: float = 0.25,  ## warm_up epoch/all epoch
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:

    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)    
    input = torch.rand(batch_size, *input_size, device=device)    
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():   
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()
                model(input)
                total += batch_size  

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start   

    throughput = total / elapsed   ## 吞吐

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]
