from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import tree
from einops import rearrange
from torch.nn import functional as F
import pdb

from rt1_pytorch.rt1_model import RT1Model
from rt1_pytorch.tokenizers.action_tokenizer import RT1ActionTokenizer

class RT1Policy:
    def __init__(
        self,
        dist: bool,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        arch: str = "efficientnet_b3",
        action_bins=256,
        num_layers=8,
        # num_heads=8,  # CHANGED: no longer used with Mamba
        # feed_forward_size=512,  # CHANGED: not used with Mamba
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
        device="cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes the RT1Policy.

        Args:
            dist (bool): Whether there is a distance input. (Ignored in this configuration.)
            observation_space (gym.spaces.Dict): The observation space.
            action_space (gym.spaces.Dict): The action space.
            arch (str, optional): The architecture to use. Defaults to "efficientnet_b3".
            action_bins (int, optional): Number of bins for discretizing actions. Defaults to 256.
            num_layers (int, optional): Number of transformer layers. Defaults to 8.
            num_heads (int, optional): Number of attention heads. Defaults to 8.  # CHANGED: Unused with Mamba
            feed_forward_size (int, optional): Feedforward layer size. Defaults to 512.  # CHANGED: Unused with Mamba
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            time_sequence_length (int, optional): The length of the time sequence. Defaults to 6.
            embedding_dim (int, optional): Embedding dimension. Defaults to 512.
            use_token_learner (bool, optional): Whether to use token learner. Defaults to True.
            token_learner_bottleneck_dim (int, optional): Bottleneck dimension for token learner. Defaults to 64.
            token_learner_num_output_tokens (int, optional): Number of output tokens from token learner. Defaults to 8.
            device (str, optional): Device for model computations. Defaults to "cuda".
            checkpoint_path (str, optional): Path to a checkpoint to load. Defaults to None.
        """
        self.dist = dist
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space,
            action_bins=action_bins,
            action_order=list(action_space.keys()),
        )

        # CHANGED: Updated RT1Model instantiation to remove unused num_heads parameter.
        # breakpoint()
        self.model = RT1Model(
            dist=self.dist,
            arch=arch,
            tokens_per_image=8,#self.action_tokenizer.tokens_per_action,
            action_bins=action_bins,
            num_layers=num_layers,
            # num_heads=num_heads,  # DELETED: Removed for Mamba usage
            # feed_forward_size=feed_forward_size,  # CHANGED: Included for legacy; not used with Mamba
            dropout_rate=dropout_rate,
            time_sequence_length=time_sequence_length,
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
            device=device,
        )

        self.embedding_dim = embedding_dim

        for action_space in self.action_space.values():
            if (
                isinstance(action_space, gym.spaces.Discrete)
                and action_space.n == time_sequence_length
            ):
                raise ValueError(
                    f"""stupid hack:Time sequence length ({time_sequence_length}) 
                    must be different from action space length ({action_space.n})."""
                )

        self.device = device
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}...")
            if self.dist:
                state_dict = torch.load(checkpoint_path)
                load_result = self.model.load_state_dict(state_dict, strict=False)
                # print("Missing keys:", load_result.missing_keys)
                # print("Unexpected keys:", load_result.unexpected_keys)
            else:
                self.model.load_state_dict(torch.load(checkpoint_path))

    def preprocess(
        self,
        videos: Union[np.ndarray, List[np.ndarray]],
        texts: Union[np.ndarray, List[np.ndarray]],
        ee_obj_dist: torch.Tensor,  # CHANGED: Distance input (unused when dist is False)
        goal_dist: torch.Tensor,    # CHANGED: Distance input (unused when dist is False)
        actions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocesses the input videos, texts, distances, and actions.

        Args:
            videos (Union[np.ndarray, List[np.ndarray]]): Input videos.
            texts (Union[np.ndarray, List[np.ndarray]]): Input texts.
            ee_obj_dist (torch.Tensor): End-effector to object distances.
            goal_dist (torch.Tensor): Base to goal distances.
            actions (Optional[Dict]): Input actions.

        Returns:
            Tuple containing preprocessed videos, texts, ee_obj_dist, goal_dist, and actions.
        """
        # CHANGED: Distance handling is not used when self.dist is False.
        # if self.dist:
        #     if isinstance(ee_obj_dist, torch.Tensor):
        #         ee_obj_dist = ee_obj_dist.to(self.device)
        #     else:
        #         ee_obj_dist = torch.tensor(ee_obj_dist, device=self.device, dtype=torch.float32)
        #
        #     if isinstance(goal_dist, torch.Tensor):
        #         goal_dist = goal_dist.to(self.device)
        #     else:
        #         goal_dist = torch.tensor(goal_dist, device=self.device, dtype=torch.float32)

        if isinstance(videos, torch.Tensor):
            videos = videos.to(self.device)
        elif not isinstance(videos, np.ndarray):
            videos = np.stack(videos, axis=0)
        
        if not isinstance(videos, torch.Tensor):
            videos = torch.tensor(videos, device=self.device, dtype=torch.float32)

        if isinstance(texts, torch.Tensor):
            texts = texts.to(self.device)
        elif not isinstance(texts, np.ndarray):
            texts = np.stack(texts, axis=0)
        if not isinstance(texts, torch.Tensor):
            texts = torch.tensor(texts, device=self.device, dtype=torch.float32)

        if actions is not None:
            actions = {
                k: np.stack(v, axis=0) if not (isinstance(v, np.ndarray)) else v
                for k, v in actions.items()
            }
            actions = tree.map_structure(
                lambda a: rearrange(a, "b f ... -> (b f) ..."), actions
            )
            actions = self.action_tokenizer.tokenize(actions)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            actions = rearrange(actions, "(b f) ... -> b f ...", b=videos.shape[0])

        # CHANGED: When self.dist is False, we return without processing distance inputs.
        if self.dist:
            return videos, texts, ee_obj_dist, goal_dist, actions  # CHANGED
        else:
            return videos, texts, actions  # CHANGED

    def forward(
        self,
        videos: torch.Tensor,
        texts: torch.Tensor,
        ee_obj_dist: torch.Tensor,  # CHANGED: Unused when dist is False
        goal_dist: torch.Tensor,    # CHANGED: Unused when dist is False
        action_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            videos (torch.Tensor): Input videos.
            texts (torch.Tensor): Input texts.
            ee_obj_dist (torch.Tensor): End-effector to object distances.
            goal_dist (torch.Tensor): Base to goal distances.
            action_logits (Optional[torch.Tensor]): Optional input action logits.

        Returns:
            Tuple containing sampled actions and action logits.
        """
        action_logits = self.model(ee_obj_dist, goal_dist, videos, texts, action_logits)  # CHANGED
        actions = torch.distributions.Categorical(logits=action_logits)
        actions = actions.sample()
        return actions, action_logits

    def loss(self, observations: Dict, target_actions: Dict) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            observations (Dict): Dictionary with keys "image", "context", "ee_obj_dist", "goal_dist".
            target_actions (Dict): Dictionary with target actions.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.dist:
            ee_obj_dist = observations["ee_obj_dist"]  # CHANGED
            goal_dist = observations["goal_dist"]      # CHANGED
        else:
            ee_obj_dist = None  # CHANGED
            goal_dist = None  # CHANGED

        videos = observations["image"]
        texts = observations["context"]
        if self.dist:
            videos, texts, ee_obj_dist, goal_dist, target_actions = self.preprocess(
                videos, texts, ee_obj_dist, goal_dist, target_actions  # CHANGED
            )
        else:
            videos, texts, target_actions = self.preprocess(
                videos, texts, ee_obj_dist, goal_dist, target_actions  # CHANGED
            )

        _, action_logits = self.forward(videos, texts, ee_obj_dist, goal_dist)  # CHANGED

        action_logits = rearrange(action_logits, "b f a d -> (b f a) d")
        target_actions = rearrange(target_actions, "b f a -> (b f a)")
        loss = F.cross_entropy(action_logits, target_actions, reduction="mean")

        dummy_loss = F.cross_entropy(action_logits, target_actions, reduction="none")
        loss_std = torch.std(dummy_loss)

        return loss, loss_std

    def act(self, observations: Dict) -> Dict[str, np.ndarray]:
        """
        Determines an action based on observations.

        Args:
            observations (Dict): Dictionary with keys "image", "context", "ee_obj_dist", "goal_dist".

        Returns:
            Dict[str, np.ndarray]: Dictionary with the key "actions".
        """
        if self.dist:
            ee_obj_dist = observations['ee_obj_dist']  # CHANGED
            goal_dist = observations['goal_dist']      # CHANGED
        videos = observations["image"]
        texts = observations["context"]

        if self.dist:
            videos, texts, ee_obj_dist, goal_dist, _ = self.preprocess(videos, texts, ee_obj_dist, goal_dist)  # CHANGED
        else:
            videos, texts, _ = self.preprocess(videos, texts, None, None)  # CHANGED

        with torch.no_grad():
            if self.dist:
                actions, _ = self.forward(videos, texts, ee_obj_dist, goal_dist)  # CHANGED
            else:
                actions, _ = self.forward(videos, texts, None, None)  # CHANGED

        actions = actions.detach().cpu().numpy()
        actions = self.action_tokenizer.detokenize(actions)
        actions = tree.map_structure(lambda a: a[:, -1], actions)
        return actions