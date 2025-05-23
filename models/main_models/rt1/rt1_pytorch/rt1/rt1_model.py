from typing import Optional

import torch
from einops import rearrange
from torch import nn
#from sklearn.preprocessing import MinMaxScaler
from numpy import array

from rt1_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer


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
        torch.Tensor: The positional embeddings of shape (seq, dim), with each element computed as the concatenation of the sine and cosine values.

    """
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# Robotic Transformer
class RT1Model(nn.Module):
    def __init__(
        self,
        dist: bool,
        arch: str = "efficientnet_b3",
        tokens_per_action=6,
        action_bins=256,
        num_layers=8,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
        device="cuda",
    ):
        """
        Initializes the RT1Model.

        Parameters:
            dist (bool): whether or not there is a distance input. True is there is, False is there isn't.
            arch (str): The efficientnet variant to use. Default is "efficientnet_b3".
            tokens_per_action (int): The number of tokens per action. Default is 11.
            action_bins (int): The number of action bins. Default is 256.
            num_layers (int): The number of transformer layers. Default is 6.
            num_heads (int): The number of attention heads. Default is 8.
            feed_forward_size (int): The size of the feed-forward layer. Default is 512.
            dropout_rate (float): The dropout rate. Default is 0.1.
            time_sequence_length (int): The length of the time sequence. Default is 6.
            embedding_dim (int): The dimension of the embedding. Default is 512.
            use_token_learner (bool): Whether to use token learner. Default is True.
            token_learner_bottleneck_dim (int): The dimension of the token learner bottleneck. Default is 64.
            token_learner_num_output_tokens (int): The number of output tokens of the token learner. Default is 8.
            device (torch.device, optional): The device for tensor operations. Defaults to "cuda".

        Returns:
            None
        """
        super().__init__()
        self.dist = dist
        if self.dist:
            self.obj_dist_encoder = nn.Linear(1, embedding_dim, device=device) #added
            self.goal_dist_encoder = nn.Linear(1, embedding_dim, device=device) #added2
        self.time_sequence_length = time_sequence_length
        self.action_encoder = nn.Linear(action_bins, embedding_dim, device=device)
        self.image_tokenizer = RT1ImageTokenizer(
            arch=arch,
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
            dropout_rate=dropout_rate,
            device=device,
        )

        self.num_tokens = self.image_tokenizer.num_output_tokens

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feed_forward_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            device=device,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, action_bins),
        ).to(device)

        self.tokens_per_action = tokens_per_action
        self.action_bins = action_bins
        self.embedding_dim = embedding_dim
        self.device = device

    def forward(
        self,
        ee_obj_dist: torch.Tensor, #added
        goal_dist: torch.Tensor, #added2
        videos: torch.Tensor,
        texts: Optional[torch.Tensor] = None,
        action_logits: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            ee_obj_dist (torch.Tensor): The distances between the end-effector and the target object.
                Shape is (b, f).
            goal_dist (torch.Tensor): The distances between the base and the goal.
                Shape is (b, f).
            videos (torch.Tensor): The input videos.
              Shape is (b, f, h, w, c) or (b, f, c, h, w).
            texts (Optional[torch.Tensor]): The input text embedding.
              Shape is (b, f, embedding_dim).
            action_logits (Optional[torch.Tensor]): The input action_logits.
              Shape is (b, f, tokens_per_action, action_bins).

        Returns:
            torch.Tensor: The output logits.
              Shape is (b, f, tokens_per_action, action_bins).
        """
        b, f, *_ = videos.shape
        assert (
            f == self.time_sequence_length
        ), f"Expected {self.time_sequence_length} frames, got videos.shape[1] = {f}"

        if texts is None:
            texts = torch.zeros((b, f, self.embedding_dim), device=self.device)
        if action_logits is None:
            action_logits = torch.zeros(
                (b, f, self.tokens_per_action, self.action_bins), device=self.device
            )
        elif action_logits.shape != (b, f, self.tokens_per_action, self.action_bins):
            raise ValueError(
                f"""Expected action_logits.shape = (b, f, tokens_per_action, action_bins),
                got {action_logits.shape}; did you pass in raw actions instead?"""
            )

        # pack time dimension into batch dimension
        videos = rearrange(videos, "b f ... -> (b f) ...")
        texts = rearrange(texts, "b f d -> (b f) d")
        

        if self.dist:
            # normalize the ee to obj dist values across each window
            # ee_obj_dist = torch.tensor(MinMaxScaler().fit_transform(ee_obj_dist.cpu().numpy()), device=self.device)
            # ee_obj_dist = torch.tensor(array([MinMaxScaler().fit_transform(seq.reshape(-1, 1)).flatten() for seq in ee_obj_dist.cpu().numpy()]), device=ee_obj_dist.device)
            
            # Encode the ee to obj distances
            ee_obj_dist = rearrange(ee_obj_dist, 'b f -> b f 1') #added
            ee_obj_dist_token = self.obj_dist_encoder(ee_obj_dist.float())  # Shape: (b, f, embedding_dim) #added

            # Encode the goal to obj distances
            goal_dist = rearrange(goal_dist, 'b f -> b f 1') #added2
            goal_dist_token = self.goal_dist_encoder(goal_dist.float())  # Shape: (b, f, embedding_dim) #added2

        # tokenize images and texts
        tokens = self.image_tokenizer(videos, texts)

        # unpack time dimension from batch dimension
        tokens = rearrange(tokens, "(b f) c n -> b f c n", b=b, f=f)

        # pack time dimension into token dimension
        tokens = rearrange(tokens, "b f c n -> b (f n) c")

        if self.dist:
            tokens = torch.cat([tokens, ee_obj_dist_token, goal_dist_token], dim=1)  # Concatenates along the sequence dimension #added, added2
        else:
            action_logits = rearrange(action_logits, "b f a d -> b (f a) d")


        # sinusoidal positional embedding
        pos_emb = posemb_sincos_1d(tokens.shape[1], tokens.shape[2], device=self.device)
        tokens = tokens + pos_emb

        # causal mask for tokens
        token_mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool).tril(0)
        token_mask = ~token_mask
        token_mask = token_mask.bool()
        token_mask = token_mask.to(self.device)

        # Rearrange to match the tokens' shape
        if self.dist:
            action_logits = rearrange(action_logits, "b f a d -> b (f a) d")
        # encode action_logits to have the same embedding dimension as tokens
        action_tokens = self.action_encoder(action_logits)

        pos_emb = posemb_sincos_1d(action_tokens.shape[1], action_tokens.shape[2], device=self.device)
        action_tokens = action_tokens + pos_emb

        # action mask: do not let action_logits attend to previous action_logits,
        # a_t is independent of a_{t-1} given pi and s_t
        action_mask = torch.ones(self.time_sequence_length, self.time_sequence_length, dtype=torch.bool).tril(0)
        action_mask = torch.kron(
            torch.eye(self.tokens_per_action, self.tokens_per_action, dtype=torch.bool),
            action_mask,
        ).bool()
        action_mask = ~action_mask
        action_mask = action_mask.bool()  # Ensure dtype is torch.bool
        action_mask = action_mask.to(self.device)

        # causal mask between tokens and action_logits;
        # a_t attends to s_t' for all t'<=t
        memory_mask = torch.ones(self.time_sequence_length, self.time_sequence_length, dtype=torch.bool).tril(0)
        if self.dist:
            memory_mask = torch.kron(
                memory_mask,
                torch.ones(self.tokens_per_action, self.num_tokens+2, dtype=torch.bool), #added, added2
            ).bool()
        else:
            memory_mask = torch.kron(
                memory_mask,
                torch.ones(self.tokens_per_action, self.num_tokens, dtype=torch.bool),
            ).bool()
        memory_mask = ~memory_mask
        memory__mask = memory_mask.bool()
        memory_mask = memory_mask.to(self.device)

        attended_tokens = self.transformer(
            src=tokens,
            src_mask=token_mask,
            tgt=action_tokens,
            tgt_mask=action_mask,
            memory_mask=memory_mask,
        )

        # unpack time dimension from token dimension
        attended_tokens = rearrange(attended_tokens, "b (f n) c -> b f n c", b=b, f=f)

        logits = self.to_logits(attended_tokens)
        return logits