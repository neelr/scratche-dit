import torch as T
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, latent_dim):
        super(LayerNorm, self).__init__()
        self.mean = nn.Parameter(T.zeros(latent_dim))
        self.sd = nn.Parameter(T.ones(latent_dim))

    def forward(self, x):
        return x * self.sd + self.mean


class AdaLN(nn.Module):
    def __init__(self, latent_dim, out_dim=None):
        super(AdaLN, self).__init__()
        out_dim = latent_dim if out_dim == None else out_dim
        self.mean = nn.Linear(latent_dim, out_dim)
        self.sd = nn.Linear(latent_dim, out_dim)
        self.ln = LayerNorm(out_dim)

    def forward(self, x, t, c):
        t = F.silu(t + c)
        mean = self.mean(t)
        sd = self.sd(t)

        # mean = T.clip(mean, max=3)
        # sd = T.clip(sd, max=3)
        return self.ln(x*sd + mean)


class Transformer(nn.Module):
    def __init__(self, seq_len, heads, latent_dim, dim_out,  steps=21, classes=10, mask=None, dropout=0.05):
        super(Transformer, self).__init__()
        self.proj = nn.Linear(seq_len, latent_dim)
        self.proj_q = nn.Linear(latent_dim, latent_dim)
        self.proj_k = nn.Linear(latent_dim, latent_dim)
        self.proj_v = nn.Linear(latent_dim, latent_dim)

        self.layer_norm1 = AdaLN(latent_dim)
        self.layer_norm2 = AdaLN(latent_dim)

        self.mask = mask
        self.seq_len = seq_len
        self.heads = heads
        self.latent_dim = latent_dim

        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, dim_out * 4),
            nn.GELU(),
            nn.Linear(dim_out * 4, dim_out),
        )

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, t, c):
        x = x  # (bs, emb, seq_len)
        x = self.proj(x)  # (bs, emb, seq_len) => (bs, emb, latent)
        bs = x.shape[0]
        emb_size = x.shape[1]
        res = x
        x = self.layer_norm1(x, t, c)

        q, k, v = [layer(x).reshape(bs, -1, self.heads, self.latent_dim//self.heads)
                   for layer in [self.proj_q, self.proj_k, self.proj_v]]
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=False)  # ,attn_mask=self.mask)
        attn = attn.reshape(bs, self.latent_dim, emb_size).permute(0, 2, 1)
        x = res + attn

        x = self.layer_norm2(x, t, c)
        return self.ffn(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size, d_model):
        super(PositionalEmbedding, self).__init__()
        dims = [T.arange(0, seq_len)/10000**(i/d_model)
                for i in range(patch_size**2)]
        dims = T.stack(dims, dim=0).to(mps)

        dims[0::2] = T.sin(dims[0::2])
        dims[1::2] = T.cos(dims[1::2])

        self.pos = dims

    def forward(self, x):
        return x + self.pos


class DiT(nn.Module):
    def __init__(self, layers, seq_len, patch_size, n_heads, steps=21, classes=10, latent_dim=None):
        super(DiT, self).__init__()
        assert (seq_len**0.5).is_integer()

        self.latent_dim = latent_dim if latent_dim != None else seq_len // 2
        self.size = int(patch_size * seq_len**0.5)
        self.patch_size = patch_size
        self.seq_len = seq_len

        self.emb = PositionalEmbedding(seq_len, patch_size, 3)

        self.transformer_block = nn.ModuleList(
            [Transformer(seq_len, n_heads, latent_dim, seq_len,
                         steps=steps, classes=classes) for _ in range(layers)]
        )

        self.layer_norm = AdaLN(latent_dim, seq_len)

        self.proj_out = nn.Linear(seq_len, seq_len)

        self.emb_step = nn.Embedding(steps, latent_dim)
        self.emb_time = nn.Embedding(classes, latent_dim)

    def forward(self, x, t, c):
        bs = x.shape[0]
        x = F.unfold(x, kernel_size=(self.patch_size,
                     self.patch_size), stride=self.patch_size)
        x = self.emb(x)

        t = self.emb_step(t)
        c = self.emb_time(c)

        for layer in self.transformer_block:
            x = layer(x, t, c)

        x = self.layer_norm(x, t, c)
        x = self.proj_out(x)
        return F.fold(x, output_size=(self.size, self.size), kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
