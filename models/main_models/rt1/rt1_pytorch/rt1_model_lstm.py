from typing import Optional

import torch
from einops import rearrange
from torch import nn
#from sklearn.preprocessing import MinMaxScaler
from numpy import array

from rt1_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer


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
        token_learner_num_output_tokens=6,
        device="cuda",
    ):
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

        # Area 1: Transformer initialization (replace with LSTM)
        # self.transformer = nn.Transformer(
        #     d_model=embedding_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers,
        #     dim_feedforward=feed_forward_size,
        #     dropout=dropout_rate,
        #     activation="gelu",
        #     batch_first=True,
        #     device=device,
        # )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=feed_forward_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        ).to(device)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(feed_forward_size),
            nn.Linear(feed_forward_size, action_bins),
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

        # Area 2: Positional embeddings (replace/remove)
        # pos_emb = posemb_sincos_1d(tokens.shape[1], tokens.shape[2], device=self.device)
        # tokens = tokens + pos_emb
        
        # Positional embeddings removed for LSTM

        # Area 3: Causal mask (remove for LSTM)
        # token_mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool).tril(0)
        # token_mask = ~token_mask
        # token_mask = token_mask.bool()
        # token_mask = token_mask.to(self.device)

        # token_mask removed for LSTM, as it is not needed

        # Area 4: Transformer forward pass (replace with LSTM forward logic)
        # attended_tokens = self.transformer(
        #     src=tokens,
        #     src_mask=token_mask,
        #     tgt=action_tokens,
        #     tgt_mask=action_mask,
        #     memory_mask=memory_mask,
        # )
        lstm_out, _ = self.lstm(tokens)

        # unpack time dimension from token dimension
        attended_tokens = rearrange(lstm_out, "b (f n) c -> b f n c", b=b, f=f)

        logits = self.to_logits(attended_tokens)
        return logits