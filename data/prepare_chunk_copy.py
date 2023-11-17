import argparse
import os
import random
import pickle
import numpy as np

def randomize_string_exclusive(s, min_chunk, max_chunk):
    chunks = []
    while s:
        chunk_size = random.randint(min_chunk, min(max_chunk, len(s)))
        chunks.append(s[:chunk_size])
        s = s[chunk_size:]

    random.shuffle(chunks)
    return ''.join(chunks)

def randomize_string(s, min_chunk, max_chunk, L=None):
    if L == None: L = len(s)
    chunks = []
    assert min_chunk <= max_chunk
    assert max_chunk <= len(s)
    cur = 0
    while cur < L:
        chunk_size = random.randint(min_chunk, min(max_chunk, L - cur))
        start = random.randint(0, len(s) - chunk_size - 1)
        chunks.append(s[start:start+chunk_size])
        cur += chunk_size

    res = ''.join(chunks)
    assert len(res) == L, (len(res), L)
    return res

def replace_chars_with_previous(s, p):
    """
    For each character c_i in the string s, replace it with c_{i-2} with probability p.
    
    :param s: The input string.
    :param p: The probability of replacement.
    :return: The modified string.
    """
    result = []
    for i in range(len(s)):
        # Choose the current or the i-2 character based on probability p
        if i > 1 and random.random() < p:
            result.append(s[i-2])
        else:
            result.append(s[i])
    return ''.join(result)

def generate_and_save_data(args):
    N, L = args.N, args.L
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Validation checks
    if N % (2 * L) != 0:
        raise ValueError("Ensure that 2L divides N")

    # Create a mapping from characters to integers
    with open(args.input_file, 'r') as f:
        src = f.read()
    print(f"Found {len(src)} characters")
    basic_chars = 'abcdefghijklmnopqrstuvwxyz'
    chars = sorted(list(set(basic_chars + src)))
    print(f"{len(chars)} chars= {chars}")
    assert len(src) >= N, "Input data is too small"
    # else:
    #     chars = 'abcdefghijklmnopqrstuvwxyz'
    #     src = ''.join(random.choices(chars, k=N))
    #     if args.trigram_frac:
    #         src = replace_chars_with_previous(src, args.trigram_frac)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Data generation
    data = []
    for block_idx in range(N // (2 * L)):
        # Generate first block of L characters
        if random.random() < args.input_file_frac:
            block = src[block_idx * L: (block_idx + 1) * L]
        else:
            block = ''.join(random.choices(basic_chars, k=L))
        if args.trigram_frac:
            block = replace_chars_with_previous(block, args.trigram_frac)

        if args.sentinel_token:
            block = itos[args.sentinel_token] + block[1:]

        # Splitting into chunks and permuting
        permuted_block = randomize_string(block, args.min_chunk_size, args.max_chunk_size)
        data.append(block + permuted_block)

    
    # Split into train and validation sets
    train_data = data[:int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9):]

    # concatenate and tokenize
    train_ids = [stoi[c] for c in ''.join(train_data)]
    val_ids = [stoi[c] for c in ''.join(val_data)]
    
    print(train_ids[:2*L])
    print(''.join(itos[i] for i in train_ids[:2*L]))
    # Saving data in binary format
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(output_dir, 'train.bin'))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(output_dir, 'val.bin'))
    # print(val_ids)

    # Save the meta information
    meta = {
        'vocab_size': len(chars),
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    with open(os.path.join(output_dir, 'args.txt'), 'w') as argsfile:
        for key, value in vars(args).items():
            argsfile.write(f'{key}: {value}\n')
    print(f"Data generated and saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Script")
    parser.add_argument('--N', type=int, default=2**26, help="Total number of tokens (should be a multiple of 2L)")
    parser.add_argument('--L', type=int, default=512, help="Length of each block")
    parser.add_argument('--min_chunk_size', type=int, default=1, help="Size of each chunk for permutation")
    parser.add_argument('--max_chunk_size', type=int, default=32, help="Size of each chunk for permutation")

    parser.add_argument('--output_dir', type=str, default="data/random-chunks", help="Directory to save the output data")
    parser.add_argument('--input_file', type=str, default='data/enwik8/enwik8.txt', help="Directory to save the output data")
    parser.add_argument('--sentinel_token', type=int, default=7, help="Always make this the first token in every batch")
    parser.add_argument('--input_file_frac', type=float, default=0, help='Add trigrams xy(x+1) with this probability')
    parser.add_argument('--trigram_frac', type=float, default=0, help='Add trigrams xy(x+1) with this probability')
    args = parser.parse_args()
    generate_and_save_data(args)
