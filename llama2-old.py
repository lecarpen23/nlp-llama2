import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import field


@dataclass
class ModelConfig:
    dim: int = 2048 # Dimension of the model
    n_layers: int = 16  # Number of layers in the transformer
    n_heads: int = 16  # Number of attention heads
    n_kv_heads: Optional[int] = field(default=None)  # Number of key-value heads (optional, defaults to n_heads)
    vocab_size: int = 25129  # Vocabulary size
    norm_eps: float = 1e-5  # Epsilon value for normalization
    ffn_dim_multiplier: float = 1.0  # Multiplier for the hidden dimension in the feedforward network (optional)
    multiple_of: int = 8  # Multiple of the hidden dimension in the feedforward network

    max_batch_size: int = 32  # Maximum batch size for training
    max_seq_len: int = 2048  # Maximum sequence length

    device: str = None  # Device to run the model on (optional)

    def __post_init__(self):
        # Set device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, device: str, config: ModelConfig) -> None:
        super(RotaryPositionEmbedding, self).__init__()
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"
        self.dim = head_dim

        # Calculate the rotation frequencies for positional embeddings
        theta_numerator = torch.arange(0, self.dim, 2, dtype=torch.float32)
        theta = 1.0 / torch.pow(10000, theta_numerator / self.dim).to(device)

        # Generate frequency values for positional embeddings
        m = torch.arange(seq_len, dtype=torch.float32).to(device)
        freqs = torch.outer(m, theta).float()

        # Register frequency values as a buffer in polar form
        self.register_buffer("freqs_complex", torch.polar(torch.ones_like(freqs), freqs))


    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        print(f'x shape in RotaryPositionEmbedding forward: {x.shape}')
        batch_size, seq_len, dim = x.shape
        #assert dim // 2 == self.dim, "dim must be twice the size of self.dim"
        print(f'self.dim: {self.dim}')
        print(f'Dim in RotaryPositionEmbedding forward: {dim}')
        #assert dim == self.dim, "dim must be equal to self.dim"

        # Reshape the input into a complex tensor for rotational operations
        # (B, SeqLen, Dim) -> (B, SeqLen, Dim // 2)
        x_complex = torch.view_as_complex(x.float().reshape(batch_size, seq_len, self.dim, 2))

        # Extract rotational frequencies for the given sequence length and start position
        # (SeqLen, Head_Dim // 2) -> (1, SeqLen, 1, Head_Dim // 2)
        freq_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(-2)

        # Apply rotational transformation to the input using frequency values
        # (B, SeqLen, H, Head_Dim // 2) * (1, SeqLen, 1, Head_Dim // 2) -> (B, SeqLen, H, Head_Dim // 2)
        x_rotated = x_complex * freq_complex

        # Convert the rotated complex tensor back to real-valued tensor
        # (B, SeqLen, H, Head_Dim // 2) -> (B, SeqLen, H , Head_Dim // 2, 2)
        x_out = torch.view_as_real(x_rotated)

        # Reshape to match the original input shape
        # (B, SeqLen, H , Head_Dim // 2, 2) -> (B, SeqLen, H, Head_Dim)
        x_out = x_out.reshape(batch_size, seq_len, -1)

        return x_out


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps  # Epsilon value for numerical stability
        self.gamma = nn.Parameter(torch.ones(dim))  # Learnable parameter for scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Input tensor of shape (Batch_Size, SeqLen, Dim)

        # Calculate the root-mean-square norm along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the root-mean-square norm and scale with gamma
        normalized_x = (x / rms) * self.gamma

        return normalized_x  # Return the normalized tensor


class SelfAttention(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        # Set the dim attribute
        self.dim = args.dim

        # Determine the number of key-value heads (defaults to n_heads if not specified)
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        # Set the number of query heads and the number of repetitions for K and V
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Calculate the head dimension
        self.head_dim = args.dim // args.n_heads

        # Linear transformations for queries, keys, values, and output
        self.Wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Initialize key and value caches with zeros
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))

        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(self.head_dim, args.max_seq_len, args.device, args)

    @staticmethod
    def repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:

        # Repeat the heads of K and V to match the number of heads in Q

        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (x[:, :, :, None, :]
                    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
                    )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        print(f'x shape in Self Attention forward: {x.shape}')
        #print(f"x in Self Attention forward: {x}")
        print(f"start_pos: {start_pos}")
        batch_size, seq_len, dim = x.shape  # (B, 1, dim)
        print(f'Dim in Self Attention forward: {dim}')
        assert dim == self.dim, "dim must be equal to self.dim"

        # # Remove extra dimension
        #x = x.squeeze(2)


        # print('---Squeeze 2---')
        # print(f'x shape in Self Attention forward: {x.shape}')



        # (B, 1, dim) -> (B, 1, n_heads_q * head_dim)
        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        # (B, 1, n_heads_q * head_dim) -> (B, 1, n_heads_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq_flat = xq.view(batch_size, seq_len, -1)
        xk_flat = xk.view(batch_size, seq_len, -1)
        
        # Apply rotary position embedding to K and V
        xq_flat = self.rope(xq_flat, start_pos)
        xk_flat = self.rope(xk_flat, start_pos)

        # Unflatten the query, key, and value tensors
        xq = xq_flat.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk_flat.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Update key and value caches
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve key and value caches
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # Repeat the heads of K and V to match the number of heads in Q
        keys = SelfAttention.repeat_heads(keys, self.n_rep) # Static method because I kept getting an error
        values = SelfAttention.repeat_heads(values, self.n_rep)

        # (B, 1, n_heads_q, head_dim) -> (B, n_heads_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, n_heads_q, 1, head_dim) * (B, n_heads_q, head_dim, SeqLen) -> (B, n_heads_q, 1, SeqLen)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, n_heads_q, 1, SeqLen) * (B, n_heads_q, SeqLen, head_dim) -> (B, n_heads_q, 1, head_dim)
        context = torch.matmul(scores, values)

        # (B, n_heads_q, 1, head_dim) -> (B, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (B, 1, head_dim) -> (B, 1, dim)
        output = self.Wo(context)

        return output


class FeedForward(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        # Calculate the hidden dimension based on the provided parameters
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        # Adjust the hidden dimension based on ffn_dim_multiplier (if provided)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Ensure hidden_dim is a multiple of args.multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Define linear layers for the feedforward network
        self.fc1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.fc3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch_Size, SeqLen, Dim)

        # Apply the first linear transformation and activation (swish)
        swish = F.silu(self.fc1(x))

        # Apply the second linear transformation
        x_V = self.fc3(swish)

        # Element-wise multiplication
        x = swish * x_V

        # Apply the third linear transformation
        x = self.fc2(x)

        return x  # Return the output


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelConfig) -> None:
        super().__init__()

        # Check if vocab_size is specified
        assert args.vocab_size != -1, "vocab_size must be specified"

        # Store model configuration and necessary parameters
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Embedding layer for token embeddings
        self.embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Create a list of transformer encoder blocks
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # Layer normalization for the output
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # Output linear layer
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        print(f'x shape in Transformer forward: {x.shape}')
        #print(f"x in Transformer forward: {x}")
        print(f"start_pos: {start_pos}")

        assert x.dim() == 2, "Input must be 2-dimensional"

        # if x.dim() == 2:
        #     x = x.unsqueeze(-1)

        # print('---Unsqueeze -1---')
        # print(f'x shape in Transformer forward: {x.shape}')
        
        # Input shape: (Batch_Size, SeqLen)

        # # Ensure seq_len is 1
        # assert x.shape[1] == 1, "seq_len must be 1"

        # Embedding lookup
        x = self.embeddings(x)

        # Pass through each transformer encoder block
        for layer in self.layers:
            x = layer(x, start_pos)

        # Layer normalization
        x = self.norm(x)

        # Output prediction
        x = self.output(x)

        return x  # Return the output