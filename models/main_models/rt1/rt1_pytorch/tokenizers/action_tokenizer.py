# """A simple action tokenizer used with Robotics Transformer 1.

# As an example, if an action is:
# {
#     'base_displacement_vector': 
#     <array([0., 0.], dtype=float32>,
#     'base_displacement_vertical_rotation': 
#     <array([0.], dtype=float32)>, 
#     'gripper_closedness_action': 
#     <array([0.], dtype=float32)>, 
#     'rotation_delta': 
#     <array([ 0.2056443 , -0.07336313,  0.01974237], dtype=float32)>, 
#     'terminate_episode': 
#     <array([2], dtype=int32)>, 
#     'world_vector': 
#     <array([ 0.06037369,  0.04524422, -0.03576406], dtype=float32)>
# }

# Then we build a sequence of tokens of length 11 [one for each dimension].
# The int32 type action dimensions are already tokenized,
# the float dimensions are bucketed according to the spaces min and max. Each
# dimension has 'action_bins' buckets.

# Currently, this tokenizer assumes one action space and it is highly recommended
# to spaceify the 'action_order', i.e. the order of keys in the dict.
# Since after tokenization you lose that information, this
# will be useful for debugging. Actions may also be subselected for prediction,
# since not all actions are needed in the action_order.
# """
# from typing import Dict, Optional

# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box, Discrete
# import pdb

# class RT1ActionTokenizer:
#     """Tokenizes based on vocab size."""

#     def __init__(
#         self,
#         action_space: gym.spaces.Dict,
#         action_bins: int,
#         action_order: Optional[list[str]] = None,
#     ):
#         """Instantiates an RT1ActionTokenizer.

#         Args:
#           action_bins: Number of buckets to discretize action to.
#           action_order: Order of the action names, used to discern the order of
#             tokenized actions to detokenize and assemble back to action tensor
#         """
#         self._action_bins = action_bins
        
#         # filter the action keys
#         lanmp_keys = ['terminate_episode', 'pickup_release', 'body_position_delta', 'body_yaw_delta','body_pitch_delta','arm_position_delta','control_mode']
#         bridge_keys =  ['terminate_episode','world_vector', 'open_gripper', "rotation_delta"]
#         fractal_keys = ['terminate_episode', 'gripper_closedness_action', 'world_vector', 'base_displacement_vertical_rotation', 'rotation_delta','base_displacement_vector',]
#         jaco_keys =  ['terminate_episode','world_vector', 'gripper_closedness_action']

#         #NOTE: change both lines below to the specific dataset keys
#         action_order  = fractal_keys  #bridge_keys #jaco_keys, fractal_keys, lanmp_keys
#         action_space =  {key: action_space[key] for key in fractal_keys if key in set(action_space.keys())}
#         self._action_space = action_space
#         if action_order is None:
#             self._action_order = list(action_space.keys())
#         else:
#             for action in action_order:
#                 assert (
#                     action in action_space.keys()
#                 ), f"action: {action} not in action_space: {action_space.keys()}"
#             self._action_order = action_order
#         self._tokens_per_action = 0
#         for action in self._action_order:
#             action_shape = action_space[action].shape
#             if isinstance(action_space, gym.spaces.Box) and len(action_shape) != 1:
#                 raise ValueError(
#                     f"Only action shapes with single dimension supported, got {action_shape}"
#                 )
#             if isinstance(action_space[action], Discrete):
#                 # Int32 actions are already assumed to be tokens.
#                 self._tokens_per_action += 1
#             elif isinstance(action_space[action], Box):
#                 if len(action_shape) != 1:
#                     raise ValueError(
#                         f"Only action shapes with single dimension supported, got {action_shape}"
#                     )
#                 self._tokens_per_action += action_shape[0]
#             else:
#                 raise ValueError(
#                     f"Unsupported action space: {type(action_space[action])}"
#                 )

#         # We measure # of action tokens in two different way. One is by checking
#         # from action_order (above) and the other is by looping through the
#         # action space (below). We aseert the # of action tokens are the same
#         # calculated by these two ways. This will assure action_order is correctly
#         # configured, otherwise, it will throw an error in the assert.
#         num_action_token = 0
#         for space in action_space.values():
#             if space.dtype == np.int_:
#                 num_action_token += 1
#             else:
#                 num_action_token += space.shape[-1]
#         assert (
#             self._tokens_per_action == num_action_token
#         ), f"{self._tokens_per_action} != {num_action_token}"

#     @property
#     def tokens_per_action(self) -> int:
#         return self._tokens_per_action

#     @property
#     def action_space(self) -> gym.spaces.Dict:
#         return self._action_space

#     @property
#     def action_order(self) -> list[str]:
#         return self._action_order

#     def tokenize(self, action: Dict) -> np.ndarray:
#         """Tokenizes an action."""
        
#         action_tokens = []
#         for k in self._action_order:
#             #print("k equals " + str(k))
#             #print(action.keys())
#             #print(action)
#             act = action[k]  # a is [batch, (time), action_size]
#             space = self._action_space[k]
#             if isinstance(space, gym.spaces.Discrete):
#                 # Int32 actions are already assumed to be tokens
#                 if not (isinstance(act, np.ndarray)):
#                     act = np.array(act, dtype=np.int32)
#                 act = np.expand_dims(act, axis=-1)
#                 if not np.all(act < space.n):
#                     raise ValueError(f"Invalid action: {act} >= {space.n}")
#                 token = act
#             elif isinstance(space, gym.spaces.Box):
#                 low = space.low[0]
#                 high = space.high[0]
#                 act = np.clip(act, low, high)
#                 # Normalize the action [batch, actions_size]
#                 token = (act - low) / (high - low)
#                 # Bucket and discretize the action to action_bins, [batch, actions_size]
#                 token = (token * (self._action_bins - 1)).astype(np.int32)
#             #TODO: bridge
#             if k == 'open_gripper':
#                 token = token[:,None]
#             action_tokens.append(token)
#             #print(k, token.shape)
#         # Append all actions, [batch, (time), all_actions_size]
#         action_tokens = np.concatenate(action_tokens, axis=-1)
#         return action_tokens

#     def detokenize(self, action_tokens: np.ndarray) -> Dict:
#         """Detokenizes an action."""
#         action = {}
#         token_index = 0
#         if not action_tokens.shape[-1] == self._tokens_per_action:
#             action_tokens = action_tokens.reshape(
#                 *action_tokens.shape[:-1], self._tokens_per_action
#             )
#         for k in self._action_order:
#             space = self._action_space[k]
#             if isinstance(space, gym.spaces.Discrete):
#                 # Int32 actions are already assumed to be tokens.
#                 action[k] = action_tokens[..., token_index]
#                 # A poor model may output tokens outside the allowed range, in that case
#                 # set them to a default value, the 0 token in this case.
#                 action[k] = np.where(
#                     action[k] >= space.n, np.zeros_like(action[k]), action[k]
#                 )
#                 token_index += 1
#             elif isinstance(space, gym.spaces.Box):
#                 actions = []
#                 for _ in range(space.shape[0]):
#                     a = action_tokens[..., token_index : token_index + 1]
#                     a = a.astype(np.float32)
#                     a = a / (self._action_bins - 1)
#                     a = (a * (space.high[0] - space.low[0])) + space.low[0]
#                     actions.append(a)
#                     token_index += 1
#                 action[k] = np.concatenate(actions, axis=-1)
#         return action

"""A simple action tokenizer used with Robotics Transformer 1.

As an example, if an action is:
{
    'base_displacement_vector': 
    <array([0., 0.], dtype=float32>,
    'base_displacement_vertical_rotation': 
    <array([0.], dtype=float32)>, 
    'gripper_closedness_action': 
    <array([0.], dtype=float32)>, 
    'rotation_delta': 
    <array([ 0.2056443 , -0.07336313,  0.01974237], dtype=float32)>, 
    'terminate_episode': 
    <array([2], dtype=int32)>, 
    'world_vector': 
    <array([ 0.06037369,  0.04524422, -0.03576406], dtype=float32)>
}

Then we build a sequence of tokens of length 11 [one for each dimension].
The int32 type action dimensions are already tokenized,
the float dimensions are bucketed according to the spaces min and max. Each
dimension has 'action_bins' buckets.

Currently, this tokenizer assumes one action space and it is highly recommended
to spaceify the 'action_order', i.e. the order of keys in the dict.
Since after tokenization you lose that information, this
will be useful for debugging. Actions may also be subselected for prediction,
since not all actions are needed in the action_order.
"""
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class RT1ActionTokenizer:
    """Tokenizes based on vocab size."""

    def __init__(
        self,
        action_space: gym.spaces.Dict,
        action_bins: int,
        action_order: Optional[list[str]] = None,
    ):
        """Instantiates an RT1ActionTokenizer.

        Args:
          action_bins: Number of buckets to discretize action to.
          action_order: Order of the action names, used to discern the order of
            tokenized actions to detokenize and assemble back to action tensor
        """
        self._action_bins = action_bins
        
        # !! Uncomment when training on LAMBDA data !!
        #####################################################################################################
        # # filter the action keys
        # lanmp_keys = ['terminate_episode', 'body_yaw_delta','arm_position_delta','control_mode']
        # fractal_keys = ['terminate_episode', 'gripper_closedness_action', 'world_vector', 'base_displacement_vertical_rotation', 'rotation_delta','base_displacement_vector',]

        # #NOTE: change both lines below to the specific dataset keys
        # action_order  = lanmp_keys # fractal_keys
        # action_space =  {key: action_space[key] for key in lanmp_keys if key in set(action_space.keys())}

        ###########################################################################################################3

        self._action_space = action_space
        if action_order is None:
            self._action_order = list(action_space.keys())
        else:
            for action in action_order:
                assert (
                    action in action_space.keys()
                ), f"action: {action} not in action_space: {action_space.keys()}"
            self._action_order = action_order
        self._tokens_per_action = 0
        for action in self._action_order:
            action_shape = action_space[action].shape
            if isinstance(action_space, gym.spaces.Box) and len(action_shape) != 1:
                raise ValueError(
                    f"Only action shapes with single dimension supported, got {action_shape}"
                )
            if isinstance(action_space[action], Discrete):
                # Int32 actions are already assumed to be tokens.
                self._tokens_per_action += 1
            elif isinstance(action_space[action], Box):
                if len(action_shape) != 1:
                    raise ValueError(
                        f"Only action shapes with single dimension supported, got {action_shape}"
                    )
                self._tokens_per_action += action_shape[0]
            else:
                raise ValueError(
                    f"Unsupported action space: {type(action_space[action])}"
                )

        # We measure # of action tokens in two different way. One is by checking
        # from action_order (above) and the other is by looping through the
        # action space (below). We aseert the # of action tokens are the same
        # calculated by these two ways. This will assure action_order is correctly
        # configured, otherwise, it will throw an error in the assert.
        num_action_token = 0
        for space in action_space.values():
            if space.dtype == np.int_:
                num_action_token += 1
            else:
                num_action_token += space.shape[-1]
        assert (
            self._tokens_per_action == num_action_token
        ), f"{self._tokens_per_action} != {num_action_token}"

    @property
    def tokens_per_action(self) -> int:
        return self._tokens_per_action

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    @property
    def action_order(self) -> list[str]:
        return self._action_order

    def tokenize(self, action: Dict) -> np.ndarray:
        """Tokenizes an action."""
        action_tokens = []
        for k in self._action_order:
            a = action[k]  # a is [batch, (time), action_size]
            space = self._action_space[k]
            if isinstance(space, gym.spaces.Discrete):
                # Int32 actions are already assumed to be tokens
                if not (isinstance(a, np.ndarray)):
                    a = np.array(a, dtype=np.int32)
                a = np.expand_dims(a, axis=-1)
                if not np.all(a < space.n):
                    raise ValueError(f"Invalid action: {a} >= {space.n}")
                token = a
            elif isinstance(space, gym.spaces.Box):
                low = space.low[0]
                high = space.high[0]
                a = np.clip(a, low, high)
                # Normalize the action [batch, actions_size]
                token = (a - low) / (high - low)
                # Bucket and discretize the action to action_bins, [batch, actions_size]
                token = (token * (self._action_bins - 1)).astype(np.int32)
            action_tokens.append(token)
        # Append all actions, [batch, (time), all_actions_size]
        action_tokens = np.concatenate(action_tokens, axis=-1)
        return action_tokens

    def detokenize(self, action_tokens: np.ndarray) -> Dict:
        """Detokenizes an action."""
        action = {}
        token_index = 0
        if not action_tokens.shape[-1] == self._tokens_per_action:
            action_tokens = action_tokens.reshape(
                *action_tokens.shape[:-1], self._tokens_per_action
            )
        for k in self._action_order:
            space = self._action_space[k]
            if isinstance(space, gym.spaces.Discrete):
                # Int32 actions are already assumed to be tokens.
                action[k] = action_tokens[..., token_index]
                # A poor model may output tokens outside the allowed range, in that case
                # set them to a default value, the 0 token in this case.
                action[k] = np.where(
                    action[k] >= space.n, np.zeros_like(action[k]), action[k]
                )
                token_index += 1
            elif isinstance(space, gym.spaces.Box):
                actions = []
                for _ in range(space.shape[0]):
                    a = action_tokens[..., token_index : token_index + 1]
                    a = a.astype(np.float32)
                    a = a / (self._action_bins - 1)
                    a = (a * (space.high[0] - space.low[0])) + space.low[0]
                    actions.append(a)
                    token_index += 1
                action[k] = np.concatenate(actions, axis=-1)
        return action