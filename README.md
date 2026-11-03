# nanogpt

A character-level decoder-only Transformer written from scratch in PyTorch,
following Andrej Karpathy's nanoGPT series. Six blocks, four attention heads,
and a 32-dimensional embedding: small enough to train on a laptop, complete
enough to generate Shakespeare-shaped text.

## What it does

- Implements multi-head causal self-attention by hand, including the masked
  attention weights that keep the model autoregressive.
- Stacks six residual blocks over the embedding, with layer norm and dropout at
  0.4.
- Trains on any plain text file, splitting into train and validation, optimising
  with AdamW, and printing evaluation loss on both splits as it goes.
- Serialises the character vocabulary to `meta.pkl` and the weights to
  `model.pth`, then samples new text from a blank prompt.

## Run it

```bash
git clone https://github.com/abderrahmenex86/nanogpt
cd nanogpt
pip install -r requirements.txt

# put a text file named input.txt in the repo root
python train.py       # 100,000 iterations by default
python generate.py    # samples 500 characters
```

A CUDA GPU helps but is not required; the code falls back to CPU.

## Sample output

After 10,000 iterations on the Tiny Shakespeare corpus:

```text
LEONTES:
A lown hought of Richild:
I dish the pall hearful Colouse show,
But stanceford the must coss fair combitith: have ron, time of proud'd forth a mey;
```

The words are invented and the grammar is inconsistent. That is what a 32-dim
embedding and 10k iterations buy you.

## Stack

Python 3.10+ · PyTorch · tqdm · pickle for the vocabulary
