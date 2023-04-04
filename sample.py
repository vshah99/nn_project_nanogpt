"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from char_freq import get_freq_mapping, CaseInsensitiveDict
import matplotlib.pyplot as plt
from numpy import corrcoef

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/Users/vedantsh/PycharmProjects/nanoGPT/out-shakespeare-char' # ignored if init_from is not 'resume'
start = "JULIET:" + "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
#max_new_tokens = 2 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
def generate_sample_cov(verbose=True, max_new_tokens=1, plots=True):
    new_tokens = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                ystr = decode(y[0].tolist())
                #print(ystr)
                new = ystr[len(start):]
                assert len(new[:max_new_tokens]) == max_new_tokens
                new_tokens += [new[:max_new_tokens]]

    freq_mapping = get_freq_mapping()
    freq_mapping = CaseInsensitiveDict(freq_mapping)
    token_vals = []
    for tokens in new_tokens:
        vals = []
        for char in (tokens[0], tokens[-1]):
            vals += [freq_mapping[char]]

        token_vals += [vals]
    x_vals = []
    y_vals = []
    for t in token_vals:
        x_vals += [t[0]]
        y_vals += [t[1]]

    if verbose:
        print(freq_mapping)
        print(new_tokens)
        print(token_vals)
        print(x_vals)
        print(y_vals)

    if plots:
        fig, ax = plt.subplots()
        ax.scatter(x_vals,y_vals,s=70, alpha=0.03)
        ax.set_xlim([-1, 39])
        ax.set_ylim([-1, 39])
        ax.set_xlabel("Char 1 rank (0 = most frequent)")
        ax.set_ylabel("Char 2 rank (0 = most frequent)")
        fig.suptitle(f'Char 1 vs Char 2 value based on frequency. Conditioned on : \n "{start}"', fontsize=10)
        plt.show()

    return corrcoef(x_vals, y_vals)[0][1]

import timeit
times = []
#ks = [2, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 500]
ks = [10*x for x in range(1,21)]
print(f"ks = {ks}")
for k in ks:
    if k<2:
        continue
    starttime = timeit.default_timer()
    corr = generate_sample_cov(verbose=True, max_new_tokens=k, plots=False)
    timed = timeit.default_timer() - starttime
    times += [timed]
    print(f"k = {k}, corr = {corr}, time = {timed}")

if len(ks)>2:
    plt.plot(ks, times)
    plt.title("Runtime for different K values")
    plt.xlabel("K")
    plt.ylabel("Time (s)")
    plt.show()