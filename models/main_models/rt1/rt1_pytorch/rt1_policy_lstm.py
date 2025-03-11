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
        num_heads=8,
        feed_forward_size= 512,
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=6,
        device="cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes an instance of the class.

        Args:
            dist (bool): whether or not there is a distance input. True is there is, False is there isn't.
            observation_space (gym.spaces.Dict): The observation space of the environment.
            action_space (gym.spaces.Dict): The action space of the environment.
            arch (str, optional): The architecture of the model. Defaults to "efficientnet_b3".
            action_bins (int, optional): The number of bins for discretizing continuous action spaces. Defaults to 256.
            num_layers (int, optional): The number of transformer layers in the model. Defaults to 8.
            num_heads (int, optional): The number of attention heads in each transformer layer. Defaults to 8.
            feed_forward_size (int, optional): The size of the feed-forward layer in the transformer. Defaults to 256.
            dropout_rate (float, optional): The dropout rate for the transformer layers. Defaults to 0.1.
            time_sequence_length (int, optional): The length of the time sequence for the model. Defaults to 6.
            embedding_dim (int, optional): The dimensionality of the input embeddings. Defaults to 512.
            use_token_learner (bool, optional): Whether to use the token learner module. Defaults to True.
            token_learner_bottleneck_dim (int, optional): The dimensionality of the bottleneck layer in the token learner. Defaults to 64.
            token_learner_num_output_tokens (int, optional): The number of output tokens from the token learner. Defaults to 8.
            device (str, optional): The device to use for the model. Defaults to "cuda".
            checkpoint_path (str, optional): load checkpoint from path. Defaults to None.

        Returns:
            None
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

        self.model = RT1Model(
            dist=self.dist,
            arch=arch,
            tokens_per_action=self.action_tokenizer.tokens_per_action,
            action_bins=action_bins,
            num_layers=num_layers,
            num_heads=num_heads,  # Removed transformer-specific
            feed_forward_size=feed_forward_size,
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
        ee_obj_dist: torch.Tensor, #added
        goal_dist: torch.Tensor, #added2
        actions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: #added, added2
        """
        Preprocesses the given videos, texts, and actions.

        Args:
            videos (Union[np.ndarray, List[np.ndarray]]): The input videos to preprocess.
              shape: (b, t, c, h, w) or (b, t, h, w, c)
            texts (Union[np.ndarray, List[np.ndarray]]): The input texts to preprocess.
              shape: (b, t, d)
            ee_obj_dist (torch.Tensor): The distance between the end-effector and the target object
                shape: (b, t)
            goal_dist (torch.Tensor): The distance between the base and the goal
                shape: (b, t)
            actions (Optional[Dict]): The input actions to preprocess. Defaults to None.
              shape: (b, t, a)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the preprocessed videos, texts, obj dist, goal dist, and actions.
        """
        if self.dist:
            #added
            if isinstance(ee_obj_dist, torch.Tensor):
                ee_obj_dist = ee_obj_dist.to(self.device)
            else:
                ee_obj_dist = torch.tensor(ee_obj_dist, device=self.device, dtype=torch.float32)

            #added2
            if isinstance(goal_dist, torch.Tensor):
                goal_dist = goal_dist.to(self.device)
            else:
                goal_dist = torch.tensor(goal_dist, device=self.device, dtype=torch.float32)

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

        if self.dist:
            return videos, texts, ee_obj_dist, goal_dist, actions #added, added2
        else:
            return videos, texts, actions

    def forward(
        self,
        videos: torch.Tensor,
        texts: torch.Tensor,
        ee_obj_dist: torch.Tensor, #added
        goal_dist: torch.Tensor, #added2
        action_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:

            videos (torch.Tensor): Input videos.
            texts (torch.Tensor): input contexts.
            ee_obj_dist (torch.Tensor): input distance between end-effector and target object
            goal_dist (torch.Tensor): input distance between base and goal
            action_logits (Optional[torch.Tensor]): Optional input action logits.

        Returns:
            action_logits (Tuple[torch.Tensor, torch.Tensor]):
              A tuple containing the sampled actions and the action logits.
        """
        # Area 4: Forward logic
        # action_logits = self.model(ee_obj_dist, goal_dist, videos, texts, action_logits)  # Original Transformer logic
        lstm_out = self.model(ee_obj_dist, goal_dist, videos, texts)  # Adjusted for LSTM model
        actions = torch.distributions.Categorical(logits=lstm_out)
        actions = actions.sample()
        return actions, lstm_out

    def loss(self, observations: Dict, target_actions: Dict) -> torch.Tensor:
        """
        Calculates the loss function for the given inputs.

        Args:
            observations (Dict): A dictionary containing the observations.
                It should have the following keys:
                    - "image" (np.ndarray): The video observations.
                    - "context" (np.ndarray): The context.
                    - "ee_obj_dist" (torch.Tensor): distance between end-effector and the target object
                    - "goal_dist" (torch.Tensor): distance between base and the goal
            target_actions (Dict): A dictionary containing the target actions.

        Returns:
            torch.Tensor: The calculated loss value.

        Raises:
            None
        """
        if self.dist:
            ee_obj_dist = observations["ee_obj_dist"] #added
            goal_dist = observations["goal_dist"] #added2
        else:
            ee_obj_dist = None
            goal_dist = None
        videos = observations["image"]
        texts = observations["context"]
        if self.dist:
            videos, texts, ee_obj_dist, goal_dist, target_actions = self.preprocess( #added, added2
                videos,
                texts,
                ee_obj_dist, #added
                goal_dist, #added2
                target_actions,
            )
        else:
            videos, texts, target_actions = self.preprocess( #added, added2
                videos,
                texts,
                ee_obj_dist, #added
                goal_dist, #added2
                target_actions,
            )

        # _, action_logits = self.forward(videos, texts, ee_obj_dist, goal_dist)  # Original Transformer logic
        _, action_logits = self.forward(videos, texts, ee_obj_dist, goal_dist)  # Adjusted for LSTM model

        action_logits = rearrange(action_logits, "b f a d -> (b f a) d")
        target_actions = rearrange(target_actions, "b f a -> (b f a)")
        loss = F.cross_entropy(action_logits, target_actions, reduction="mean")
        # loss = loss / videos.shape[0]

        
        dummy_loss = F.cross_entropy(action_logits, target_actions, reduction="none")
        loss_std = torch.std(dummy_loss)

        return loss, loss_std

    def act(self, observations: Dict) -> Dict[str, np.ndarray]:
        """
        Performs an action based on the given observations.
        Note that this takes in observations of shape (b,t, ...)
        but only returns the last action for each trajectory of shape (b, ...).

        Args:
            observations (Dict): A dictionary containing the observations. It should have the following keys:
                - "image" (np.ndarray): The video observations.
                - "context" (np.ndarray): The context.
                - "ee_obj_dist" (torch.Tensor): distance between end-effector and the target object
                - "goal_dist" (torch.Tensor): distance between base and the goal

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the actions. It has the following keys:
                - "actions" (np.ndarray): The actions performed based on the observations.
        """

        if self.dist:
            ee_obj_dist = observations['ee_obj_dist'] #added
            goal_dist = observations['goal_dist'] #added2
        videos = observations["image"]
        texts = observations["context"]

        if self.dist:
            videos, texts, ee_obj_dist, goal_dist, _ = self.preprocess(videos, texts, ee_obj_dist, goal_dist) #added, added2
        else:
            videos, texts, _ = self.preprocess(videos, texts, None, None) #added, added2
        
        with torch.no_grad():
            # if self.dist:
            #     actions, _ = self.forward(videos, texts, ee_obj_dist, goal_dist)  # Original Transformer logic
            # else:
            #     actions, _ = self.forward(videos, texts, None, None)  # Original Transformer logic

            # Adjusted for LSTM model
            if self.dist:
                actions, _ = self.forward(videos, texts, ee_obj_dist, goal_dist)
            else:
                actions, _ = self.forward(videos, texts, None, None)

        actions = actions.detach().cpu().numpy()
        actions = self.action_tokenizer.detokenize(actions)
        actions = tree.map_structure(lambda a: a[:, -1], actions)
        return actions
