import pathlib
import torch
from esm import FastaBatchedDataset, pretrained
import argparse

def extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch, seq_length, repr_layers):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "log.txt", "w") as log_file:
        def log(message):
            print(message)
            log_file.write(message + '\n')

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                try:
                    log(f'Processing batch {batch_idx + 1} of {len(batches)}')

                    if torch.cuda.is_available():
                        toks = toks.to(device="cuda", non_blocking=True)

                    out = model(toks, repr_layers=repr_layers, return_contacts=False)
                    representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                    for i, label in enumerate(labels):
                        entry_id = label.split()[0]

                        filename = output_dir / f"{entry_id}.pt"
                        truncate_len = min(seq_length, len(strs[i]))

                        result = {"entry_id": entry_id}
                        result["mean_representations"] = {
                            layer: t[i, 1: truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }

                        torch.save(result, filename)
                except Exception as e:
                    log(f'Error processing batch {batch_idx + 1}: {str(e)}')


def main():
    parser = argparse.ArgumentParser(description='Compute protein embeddings.')
    parser.add_argument('--model_index', type=int, default=1,
                        help='Index of the model to use. 0 for esm2_t48_15B_UR50D, 1 for esm2_t36_3B_UR50D')
    parser.add_argument('--seq_length', type=int, default=40000, help='Sequence length for embeddings.')
    args = parser.parse_args()

    models = [{
        "model_name": 'esm2_t48_15B_UR50D',
        "repr_layers": [48],
        "output_dir": './ctx_full_esm2_15b_all_embs'
    }, {
        "model_name": "esm2_t36_3B_UR50D",
        "repr_layers": [36],
        "output_dir": './ctx_full_esm2_3b_all_embs'
    }]

    fasta_file = pathlib.Path('../../homo_sapiens.fasta')

    model = models[args.model_index]
    model_name = model["model_name"]
    repr_layers = model["repr_layers"]
    output_dir = model["output_dir"]
    extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch=args.seq_length, seq_length=args.seq_length,
                       repr_layers=repr_layers)


if __name__ == "__main__":
    main()
