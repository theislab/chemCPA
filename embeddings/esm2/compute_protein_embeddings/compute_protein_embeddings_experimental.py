import pathlib
import torch
from tqdm import tqdm

from esm import FastaBatchedDataset, pretrained
import torch.multiprocessing as mp
import functools
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)


from torch.utils.data import Sampler
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class DistributedBatchSampler(Sampler):
    def __init__(self, batches) -> None:
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.batches = batches

    def __iter__(self):
        resampled_batches = self.batches[self.rank:len(self.batches):self.num_replicas]
        return iter(resampled_batches)

    def __len__(self) -> int:
        return self.num_samples

def extract_embeddings(rank, world_size, model_name, fasta_file, output_dir, tokens_per_batch, seq_length, repr_layers):
    setup(rank, world_size)
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()


    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    distributed_batch_sampler = DistributedBatchSampler(batches)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=distributed_batch_sampler
    )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / "log.txt", "w")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True))


    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(batches)):

            try:
                log_file.write(f'Processing batch {batch_idx + 1} of {len(batches)}\n')
                toks = toks.to(device="cuda", non_blocking=True)
                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                batch_results = []
                for i, label in enumerate(labels):
                    entry_id = label.split()[0]
                    truncate_len = min(seq_length, len(strs[i]))

                    result = {"entry_id": entry_id}
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                    batch_results.append(result)

                batch_filename = output_dir / f"batch_{batch_idx + 1}.pt"
                torch.save(batch_results, batch_filename)
            except Exception as e:
                print(e)
                log_file.write(f'Error processing batch {batch_idx + 1}: {str(e)}\n')
                continue

    log_file.close()
    cleanup()

seq_length = 40000
models = [{
    "model_name": 'esm2_t48_15B_UR50D',
    "repr_layers": [48],
    "output_dir" : './ctx_full_esm2_15b_all_embs'
}, {
    "model_name" : "esm2_t36_3B_UR50D",
    "repr_layers": [36],
    "output_dir": './ctx_full_esm2_3b_all_embs'
}]

fasta_file = pathlib.Path('../../homo_sapiens.fasta')

model = models[1]
model_name = model["model_name"]
repr_layers = model["repr_layers"]
output_dir = model["output_dir"]
extract_embeddings()

if __name__ == '__main__':
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(extract_embeddings, args=(WORLD_SIZE, model_name, fasta_file, output_dir, seq_length , seq_length, repr_layers), nprocs=WORLD_SIZE, join=True)
