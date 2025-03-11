from typing import Optional
import torch
from einops import rearrange
from torch import nn
# from sklearn.preprocessing import MinMaxScaler  # DELETED: Unused in new design
from numpy import array

from rt1_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer
from mamba_ssm import Mamba

def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    """
    Generate positional embeddings using sine and cosine functions for a 1-dimensional sequence.

    Parameters:
        seq (int): The length of the sequence.
        dim (int): The dimension of the positional embeddings.
        temperature (float, optional): The temperature parameter for the sine function. Defaults to 10000.
        device (torch.device, optional): The device for tensor operations. Defaults to None.
        dtype (torch.dtype, optional): The data type of the positional embeddings. Defaults to torch.float32.

    Returns:
        torch.Tensor: The positional embeddings of shape (seq, dim).
    """
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)

# Robotic Transformer rewritten to use Mamba without concatenating action tokens.
class RT1Model(nn.Module):
    def __init__(
        self,
        dist: bool,
        arch: str = "efficientnet_b3",
        tokens_per_image=8,  # CORRECTED: Renamed parameter (tokens per image produced by TokenLearner)
        action_bins=256,
        num_layers=8,
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,  # This value should match tokens_per_image
        device="cuda",
    ):
        """
        Initializes the RT1Model using Mamba for processing visual tokens only.
        """
        super().__init__()
        self.dist = dist
        if self.dist:
            self.obj_dist_encoder = nn.Linear(1, embedding_dim, device=device)  # CORRECTED: kept for legacy support
            self.goal_dist_encoder = nn.Linear(1, embedding_dim, device=device)  # CORRECTED: kept for legacy support
        self.time_sequence_length = time_sequence_length
        self.action_encoder = nn.Linear(action_bins, embedding_dim, device=device)  # (Unused now)
        self.image_tokenizer = RT1ImageTokenizer(
            arch=arch,
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.num_tokens = self.image_tokenizer.num_output_tokens  # e.g., should be 8 tokens per image
        # CORRECTED: Ensure tokens_per_image matches the image tokenizer's output
        assert tokens_per_image == self.num_tokens, "tokens_per_image must equal the number of output tokens from the image tokenizer."

        # Instead of using a Transformer with separate target input, we stack multiple Mamba blocks.
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                Mamba(
                    d_model=embedding_dim,
                    d_state=256,      # Must be ≤ 256
                    d_conv=4,
                    expand=4,
                    device=device,
                )
            ).to(device)
            for _ in range(num_layers)
        ])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, action_bins),
        ).to(device)

        # Save tokens_per_image for later use in reshaping.
        self.tokens_per_image = tokens_per_image  # CORRECTED: Renamed internal variable.
        self.action_bins = action_bins
        self.embedding_dim = embedding_dim
        self.device = device

    def forward(
        self,
        ee_obj_dist: torch.Tensor,  # Unused since dist will be False
        goal_dist: torch.Tensor,    # Unused since dist will be False
        videos: torch.Tensor,
        texts: Optional[torch.Tensor] = None,
        action_logits: Optional[torch.Tensor] = None,  # Not used now
    ):
        """
        Forward pass of the model using Mamba for processing visual tokens.
        Produces action logits of shape (b, f, tokens_per_image, action_bins).

        Note: Distance-related inputs and ground-truth actions are not fed into the model.
        """
        b, f, *_ = videos.shape
        assert (
            f == self.time_sequence_length
        ), f"Expected {self.time_sequence_length} frames, got videos.shape[1] = {f}"

        if texts is None:
            texts = torch.zeros((b, f, self.embedding_dim), device=self.device)
        # action_logits is ignored in this version since we do not use teacher forcing

        # Rearranging visual inputs only.
        videos = rearrange(videos, "b f ... -> (b f) ...")
        texts = rearrange(texts, "b f d -> (b f) d")

        # Tokenize images and texts to obtain visual tokens.
        tokens = self.image_tokenizer(videos, texts)
        tokens = rearrange(tokens, "(b f) c n -> b f c n", b=b, f=f)
        visual_tokens = rearrange(tokens, "b f c n -> b (f n) c")  # Shape: [b, f * num_tokens, embedding_dim]

        # Add sinusoidal positional embeddings to visual tokens.
        pos_emb_vis = posemb_sincos_1d(visual_tokens.shape[1], visual_tokens.shape[2], device=self.device)
        visual_tokens = visual_tokens + pos_emb_vis

        # Process visual tokens through the stacked Mamba layers.
        x = visual_tokens
        for layer in self.mamba_layers:
            x = x + layer(x)  # Residual connection for each Mamba layer  # CORRECTED
        attended_tokens = x

        ###########################################

         # CORRECTED: Pool the attended tokens to reduce sequence length from (f * tokens_per_image) to (f * tokens_per_action).
        # Here, we use adaptive average pooling. The desired length is:
        desired_length = f * 6  # Typically 6 * 6 = 36.
        # attended_tokens has shape (b, L, embedding_dim) with L = f * tokens_per_image (e.g., 48).
        attended_tokens = torch.nn.functional.adaptive_avg_pool1d(attended_tokens.transpose(1, 2), output_size=desired_length).transpose(1, 2)  # CORRECTED

        # CORRECTED: Rearrange the attended tokens to shape: (b, f, tokens_per_action, embedding_dim)
        action_out = rearrange(attended_tokens, "b (f n) c -> b f n c", f=f, n=6)  # CORRECTED
        logits = self.to_logits(action_out)

        #########################################

        # The expected output sequence length should be f * tokens_per_image.
        # Rearrange the attended tokens to shape: (b, f, tokens_per_image, embedding_dim)
        # action_out = rearrange(attended_tokens, "b (f n) c -> b f n c", f=f, n=self.tokens_per_image)  # CORRECTED
        logits = self.to_logits(action_out)
        return logits