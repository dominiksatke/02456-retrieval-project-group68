import matplotlib.pyplot as plt


def plot_attention_grid(maps, tokens, mask, index=0, title=None, save_path=None):
    """Plot every layer x head attention map for one text of the batch.

    @param maps: list of n_layers tensors [B, n_heads, L, L] if return_attention=True
    @param tokens: list of L token strings for that text
    @param mask: [B, L] attention mask, 1 = token, 0 = padding
    @param index: which text of the batch to plot
    Returns: the matplotlib figure
    """
    n_layers = len(maps)
    n_heads = maps[0].shape[1]
    n = int(mask[index].sum())
    words = tokens[:n]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.5 * n_layers),layout='constrained', squeeze=False)
    # axes[l, h]
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            attn = maps[l][index, h][:n, :n]
            im = ax.imshow(attn.detach().numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=90)
            ax.set_yticks(range(n)); ax.set_yticklabels(words)
            ax.set_title(f'L {l+1}, H {h+1}')
    fig.supxlabel('key (looked at)')
    fig.supylabel('query (looking)')
    fig.colorbar(im, ax=axes, shrink=0.6)

    if title is not None:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_selected_heads(maps, tokens, mask, heads, index=0, title=None, save_path=None):
    """Plot a few chosen (layer, head) attention maps side by side.

    @param heads: list of (layer, head) pairs, 0-indexed
    """
    n = int(mask[index].sum())                              
    words = tokens[:n]
    size = max(4, 0.25 * n)          
    fig, axes = plt.subplots(1, len(heads), figsize=(size * len(heads), size + 0.5),
                             layout='constrained', squeeze=False)

    for i, (l, h) in enumerate(heads):
        ax = axes[0, i]
        attn = maps[l][index, h][:n, :n]
        im = ax.imshow(attn.detach().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=90)
        ax.set_yticks(range(n)); ax.set_yticklabels(words)
        ax.set_title(f'L{l+1} H{h+1}')
    fig.supxlabel('key (looked at)')
    fig.supylabel('query (looking)')
    fig.colorbar(im, ax=axes, shrink=0.8)

    if title is not None:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
