"""
FineWeb-Edu 1BT dataset downloader/tokenizer.

This is a smaller variant of get.py:
- uses the Hugging Face domestic mirror at https://hf-mirror.com
- reads mirrored FineWeb-Edu 10BT parquet files directly and stops after 1BT tokens
- writes 100M-token shards to data/edu_fineweb1B
"""

import os

HF_MIRROR = "https://hf-mirror.com"

# Must be set before importing datasets/huggingface_hub. Force override any
# existing environment value so the script never silently falls back upstream.
os.environ["HF_ENDPOINT"] = HF_MIRROR
os.environ["HF_HUB_ENDPOINT"] = HF_MIRROR

import multiprocessing as mp

import numpy as np
import tiktoken
import datasets.config
import huggingface_hub.constants
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# Some older/newer huggingface_hub + datasets combinations cache endpoints at
# import time or use different constants. Patch both libraries explicitly.
huggingface_hub.constants.ENDPOINT = HF_MIRROR
datasets.config.HF_ENDPOINT = HF_MIRROR

# ------------------------------------------
local_dir = "data/edu_fineweb1B"
sample_dir = "sample/10BT"
target_tokens = int(1e9)  # 1BT = 1 billion GPT-2 tokens
shard_size = int(1e8)  # 100M tokens per shard, 10 shards for 1BT
data_files = [
    f"{HF_MIRROR}/datasets/HuggingFaceFW/fineweb-edu/resolve/main/{sample_dir}/{i:03d}_00000.parquet"
    for i in range(13)
]

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    print(f"Using Hugging Face mirror: {HF_MIRROR}")
    print(f"First parquet file: {data_files[0]}")

    # Download parquet files directly from the domestic mirror. This avoids the
    # Hub metadata API, which may otherwise still try huggingface.co.
    fw = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True,
    )

    # tokenize documents and write output shards, stopping after target_tokens
    nprocs = max(1, os.cpu_count() // 2)
    tokens_written = 0

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            remaining_total = target_tokens - tokens_written - token_count
            if remaining_total <= 0:
                break

            if len(tokens) > remaining_total:
                tokens = tokens[:remaining_total]

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    current_shard_total = min(shard_size, target_tokens - tokens_written)
                    progress_bar = tqdm(total=current_shard_total, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                tokens_written += shard_size
                shard_index += 1
                progress_bar = None

                if tokens_written >= target_tokens:
                    token_count = 0
                    break

                # populate the next shard with the leftovers of the current doc
                leftover = len(tokens) - remainder
                all_tokens_np[0:leftover] = tokens[remainder:]
                token_count = leftover
                if token_count:
                    current_shard_total = min(shard_size, target_tokens - tokens_written)
                    progress_bar = tqdm(total=current_shard_total, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(token_count)

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            if progress_bar is not None:
                progress_bar.close()


if __name__ == "__main__":
    main()
