# Taken from https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0
import abc
import dataclasses
from typing import Any, Dict, Iterable, Optional, Union
import pdb
import numpy as np
import reverb
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from rlds import rlds_types, transformations

tf.config.experimental.set_visible_devices([], "GPU")


def dataset2path(name):
    if name == "robo_net":
        version = "1.0.0"
    elif name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{name}/{version}"


def as_gif(images, path="temp.gif"):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


def _features_to_tensor_spec(feature: tfds.features.FeatureConnector) -> tf.TensorSpec:
    """Converts a tfds Feature into a TensorSpec."""

    def _get_feature_spec(nested_feature: tfds.features.FeatureConnector):
        if isinstance(nested_feature, tf.DType):
            return tf.TensorSpec(shape=(), dtype=nested_feature)
        else:
            return nested_feature.get_tensor_spec()

    # FeaturesDict can sometimes be a plain dictionary, so we use tf.nest to
    # make sure we deal with the nested structure.
    return tf.nest.map_structure(_get_feature_spec, feature)


def _encoded_feature(
    feature: Optional[tfds.features.FeatureConnector],
    image_encoding: Optional[str],
    tensor_encoding: Optional[tfds.features.Encoding],
):
    """Adds encoding to Images and/or Tensors."""

    def _apply_encoding(
        feature: tfds.features.FeatureConnector,
        image_encoding: Optional[str],
        tensor_encoding: Optional[tfds.features.Encoding],
    ):
        if image_encoding and isinstance(feature, tfds.features.Image):
            return tfds.features.Image(
                shape=feature.shape,
                dtype=feature.dtype,
                use_colormap=feature.use_colormap,
                encoding_format=image_encoding,
            )
        if (
            tensor_encoding
            and isinstance(feature, tfds.features.Tensor)
            and feature.dtype != tf.string
        ):
            return tfds.features.Tensor(
                shape=feature.shape, dtype=feature.dtype, encoding=tensor_encoding
            )
        return feature

    if not feature:
        return None
    return tf.nest.map_structure(
        lambda x: _apply_encoding(x, image_encoding, tensor_encoding), feature
    )


@dataclasses.dataclass
class RLDSSpec(metaclass=abc.ABCMeta):
    """Specification of an RLDS Dataset.

    It is used to hold a spec that can be converted into a TFDS DatasetInfo or
    a `tf.data.Dataset` spec.
    """

    observation_info: Optional[tfds.features.FeatureConnector] = None
    action_info: Optional[tfds.features.FeatureConnector] = None
    reward_info: Optional[tfds.features.FeatureConnector] = None
    discount_info: Optional[tfds.features.FeatureConnector] = None
    step_metadata_info: Optional[tfds.features.FeaturesDict] = None
    episode_metadata_info: Optional[tfds.features.FeaturesDict] = None

    def step_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        step = {}
        if self.observation_info:
            step[rlds_types.OBSERVATION] = _features_to_tensor_spec(
                self.observation_info
            )
        if self.action_info:
            step[rlds_types.ACTION] = _features_to_tensor_spec(self.action_info)
        if self.discount_info:
            step[rlds_types.DISCOUNT] = _features_to_tensor_spec(self.discount_info)
        if self.reward_info:
            step[rlds_types.REWARD] = _features_to_tensor_spec(self.reward_info)
        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step[k] = _features_to_tensor_spec(v)

        step[rlds_types.IS_FIRST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_LAST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_TERMINAL] = tf.TensorSpec(shape=(), dtype=bool)
        return step

    def episode_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        episode = {}
        episode[rlds_types.STEPS] = tf.data.DatasetSpec(
            element_spec=self.step_tensor_spec()
        )
        if self.episode_metadata_info:
            for k, v in self.episode_metadata_info.items():
                episode[k] = _features_to_tensor_spec(v)
        return episode

    def to_dataset_config(
        self,
        name: str,
        image_encoding: Optional[str] = None,
        tensor_encoding: Optional[tfds.features.Encoding] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        description: Optional[str] = None,
        overall_description: Optional[str] = None,
    ) -> tfds.rlds.rlds_base.DatasetConfig:
        """Obtains the DatasetConfig for TFDS from the Spec."""
        return tfds.rlds.rlds_base.DatasetConfig(
            name=name,
            description=description,
            overall_description=overall_description,
            homepage=homepage,
            citation=citation,
            observation_info=_encoded_feature(
                self.observation_info, image_encoding, tensor_encoding
            ),
            action_info=_encoded_feature(
                self.action_info, image_encoding, tensor_encoding
            ),
            reward_info=_encoded_feature(
                self.reward_info, image_encoding, tensor_encoding
            ),
            discount_info=_encoded_feature(
                self.discount_info, image_encoding, tensor_encoding
            ),
            step_metadata_info=_encoded_feature(
                self.step_metadata_info, image_encoding, tensor_encoding
            ),
            episode_metadata_info=_encoded_feature(
                self.episode_metadata_info, image_encoding, tensor_encoding
            ),
        )

    def to_features_dict(self):
        """Returns a TFDS FeaturesDict representing the dataset config."""
        step_config = {
            rlds_types.IS_FIRST: tf.bool,
            rlds_types.IS_LAST: tf.bool,
            rlds_types.IS_TERMINAL: tf.bool,
        }

        if self.observation_info:
            step_config[rlds_types.OBSERVATION] = self.observation_info
        if self.action_info:
            step_config[rlds_types.ACTION] = self.action_info
        if self.discount_info:
            step_config[rlds_types.DISCOUNT] = self.discount_info
        if self.reward_info:
            step_config[rlds_types.REWARD] = self.reward_info

        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step_config[k] = v

        if self.episode_metadata_info:
            return tfds.features.FeaturesDict(
                {
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
                    **self.episode_metadata_info,
                }
            )
        else:
            return tfds.features.FeaturesDict(
                {
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
                }
            )


RLDS_SPEC = RLDSSpec
TENSOR_SPEC = Union[tf.TensorSpec, dict[str, tf.TensorSpec]]


@dataclasses.dataclass
class TrajectoryTransform(metaclass=abc.ABCMeta):
    """Specification the TrajectoryTransform applied to a dataset of episodes.

    A TrajectoryTransform is a set of rules transforming a dataset
    of RLDS episodes to a dataset of trajectories.
    This involves three distinct stages:
    - An optional `episode_to_steps_map_fn(episode)` is called at the episode
      level, and can be used to select or modify steps.
      - Augmentation: an `episode_key` could be propagated to `steps` for
        debugging.
      - Selection: Particular steps can be selected.
      - Stripping: Features can be removed from steps. Prefer using `step_map_fn`.
    - An optional `step_map_fn` is called at the flattened steps dataset for each
      step, and can be used to featurize a step, e.g. add/remove features, or
      augument images
    - A `pattern` leverages DM patterns to set a rule of slicing an episode to a
      dataset of overlapping trajectories.

    Importantly, each TrajectoryTransform must define a `expected_tensor_spec`
    which specifies a nested TensorSpec of the resulting dataset. This is what
    this TrajectoryTransform will produce, and can be used as an interface with
    a neural network.
    """

    episode_dataset_spec: RLDS_SPEC
    episode_to_steps_fn_dataset_spec: RLDS_SPEC
    steps_dataset_spec: Any
    pattern: reverb.structured_writer.Pattern
    episode_to_steps_map_fn: Any
    expected_tensor_spec: TENSOR_SPEC
    step_map_fn: Optional[Any] = None

    def get_for_cached_trajectory_transform(self):
        """Creates a copy of this traj transform to use with caching.

        The returned TrajectoryTransfrom copy will be initialized with the default
        version of the `episode_to_steps_map_fn`, because the effect of that
        function has already been materialized in the cached copy of the dataset.
        Returns:
          trajectory_transform: A copy of the TrajectoryTransform with overridden
            `episode_to_steps_map_fn`.
        """
        traj_copy = dataclasses.replace(self)
        traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
        traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
        return traj_copy

    def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
        """Applies this TrajectoryTransform to the dataset of episodes."""

        # Convert the dataset of episodes to the dataset of steps.
        steps_dataset = episodes_dataset.map(
            self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).flat_map(lambda x: x)

        return self._create_pattern_dataset(steps_dataset)

    def transform_steps_rlds_dataset(
        self, steps_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Applies this TrajectoryTransform to the dataset of episode steps."""

        return self._create_pattern_dataset(steps_dataset)

    def create_test_dataset(
        self,
    ) -> tf.data.Dataset:
        """Creates a test dataset of trajectories.

        It is guaranteed that the structure of this dataset will be the same as
        when flowing real data. Hence this is a useful construct for tests or
        initialization of JAX models.
        Returns:
          dataset: A test dataset made of zeros structurally identical to the
            target dataset of trajectories.
        """
        zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

        return tf.data.Dataset.from_tensors(zeros)

    def _create_pattern_dataset(
        self, steps_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Create PatternDataset from the `steps_dataset`."""
        config = create_structured_writer_config("temp", self.pattern)

        # Further transform each step if the `step_map_fn` is provided.
        if self.step_map_fn:
            steps_dataset = steps_dataset.map(self.step_map_fn)
        pattern_dataset = reverb.PatternDataset(
            input_dataset=steps_dataset,
            configs=[config],
            respect_episode_boundaries=True,
            is_end_of_episode=lambda x: x[rlds_types.IS_LAST],
        )
        return pattern_dataset


class TrajectoryTransformBuilder(object):
    """Facilitates creation of the `TrajectoryTransform`."""

    def __init__(
        self,
        dataset_spec: RLDS_SPEC,
        episode_to_steps_map_fn=lambda e: e[rlds_types.STEPS],
        step_map_fn=None,
        pattern_fn=None,
        expected_tensor_spec=None,
    ):
        self._rds_dataset_spec = dataset_spec
        self._steps_spec = None
        self._episode_to_steps_map_fn = episode_to_steps_map_fn
        self._step_map_fn = step_map_fn
        self._pattern_fn = pattern_fn
        self._expected_tensor_spec = expected_tensor_spec

    def build(self, validate_expected_tensor_spec: bool = True) -> TrajectoryTransform:
        """Creates `TrajectoryTransform` from a `TrajectoryTransformBuilder`."""

        if validate_expected_tensor_spec and self._expected_tensor_spec is None:
            raise ValueError("`expected_tensor_spec` must be set.")

        episode_ds = zero_episode_dataset_from_spec(self._rds_dataset_spec)

        steps_ds = episode_ds.flat_map(self._episode_to_steps_map_fn)

        episode_to_steps_fn_dataset_spec = self._rds_dataset_spec

        if self._step_map_fn is not None:
            steps_ds = steps_ds.map(self._step_map_fn)

        zeros_spec = transformations.zeros_from_spec(
            steps_ds.element_spec
        )  # pytype: disable=wrong-arg-types

        ref_step = reverb.structured_writer.create_reference_step(zeros_spec)

        pattern = self._pattern_fn(ref_step)

        steps_ds_spec = steps_ds.element_spec

        target_tensor_structure = create_reverb_table_signature(
            "temp_table", steps_ds_spec, pattern
        )

        if (
            validate_expected_tensor_spec
            and self._expected_tensor_spec != target_tensor_structure
        ):
            raise RuntimeError(
                "The tensor spec of the TrajectoryTransform doesn't "
                "match the expected spec.\n"
                "Expected:\n%s\nActual:\n%s\n"
                % (
                    str(self._expected_tensor_spec).replace(
                        "TensorSpec", "tf.TensorSpec"
                    ),
                    str(target_tensor_structure).replace("TensorSpec", "tf.TensorSpec"),
                )
            )

        return TrajectoryTransform(
            episode_dataset_spec=self._rds_dataset_spec,
            episode_to_steps_fn_dataset_spec=episode_to_steps_fn_dataset_spec,
            steps_dataset_spec=steps_ds_spec,
            pattern=pattern,
            episode_to_steps_map_fn=self._episode_to_steps_map_fn,
            step_map_fn=self._step_map_fn,
            expected_tensor_spec=target_tensor_structure,
        )


def zero_episode_dataset_from_spec(rlds_spec: RLDS_SPEC):
    """Creates a zero valued dataset of episodes for the given RLDS Spec."""

    def add_steps(episode, step_spec):
        episode[rlds_types.STEPS] = transformations.zero_dataset_like(
            tf.data.DatasetSpec(step_spec)
        )
        if "fake" in episode:
            del episode["fake"]
        return episode

    episode_without_steps_spec = {
        k: v
        for k, v in rlds_spec.episode_tensor_spec().items()
        if k != rlds_types.STEPS
    }

    if episode_without_steps_spec:
        episodes_dataset = transformations.zero_dataset_like(
            tf.data.DatasetSpec(episode_without_steps_spec)
        )
    else:
        episodes_dataset = tf.data.Dataset.from_tensors({"fake": ""})

    episodes_dataset_with_steps = episodes_dataset.map(
        lambda episode: add_steps(episode, rlds_spec.step_tensor_spec())
    )
    return episodes_dataset_with_steps


def create_reverb_table_signature(
    table_name: str, steps_dataset_spec, pattern: reverb.structured_writer.Pattern
) -> reverb.reverb_types.SpecNest:
    config = create_structured_writer_config(table_name, pattern)
    reverb_table_spec = reverb.structured_writer.infer_signature(
        [config], steps_dataset_spec
    )
    return reverb_table_spec


def create_structured_writer_config(
    table_name: str, pattern: reverb.structured_writer.Pattern
) -> Any:
    config = reverb.structured_writer.create_config(
        pattern=pattern, table=table_name, conditions=[]
    )
    return config


def n_step_pattern_builder(n: int) -> Any:
    """Creates trajectory of length `n` from all fields of a `ref_step`."""

    def transform_fn(ref_step):
        traj = {}
        for key in ref_step:
            if isinstance(ref_step[key], dict):
                transformed_entry = tree.map_structure(
                    lambda ref_node: ref_node[-n:], ref_step[key]
                )
                traj[key] = transformed_entry
            else:
                traj[key] = ref_step[key][-n:]

        return traj

    return transform_fn


def get_observation_and_action_from_step(step):
    return {
        "observation": {
            "image": step["observation"]["image"],
            "embedding": step["observation"]["natural_language_embedding"],
            "instruction": step["observation"]["natural_language_instruction"],
        },
        # Decode one hot discrete actions
        "action": {
            k: tf.argmax(v, axis=-1) if v.dtype == tf.int32 else v
            for k, v in step["action"].items()
        },
    }


def create_dataset(
    datasets=["fractal20220817_data"],
    split="train",
    trajectory_length=6,
    batch_size=32,
    num_epochs=1,
) -> Iterable[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    trajectory_datasets = []
    #tf.io.gfile.exists(builder_dir=dataset2path(dataset))
    for dataset in datasets:
        #tf.io.gfile.exists(builder_dir=dataset2path(dataset))
                
        b = tfds.builder_from_directory(builder_dir = '') #add path to fractal dataset for rt1
        # ds = tfds.load("fractal20220817_data:0.1.0", data_dir="gs://gresearch/robotics")

        ds = b.as_dataset(split=split)
    

        # Helper function to convert tensors to numpy arrays recursively
        # def convert_to_numpy(obj):
        #     if isinstance(obj, dict):
        #         return {key: convert_to_numpy(value) for key, value in obj.items()}
        #     elif isinstance(obj, tf.Tensor):
        #         return obj.numpy()
        #     elif isinstance(obj, tf.data.Dataset):  # If it's a dataset (like steps)
        #         return [convert_to_numpy(step) for step in obj.take(5)]  # Take a few steps to inspect
        #     else:
        #         return obj

        # # Iterate over the dataset and inspect a few samples
        # for example in ds.take(1):  # Take 1 example to inspect
        #     numpy_example = convert_to_numpy(example)
            
        #     # Now you can print the actual data
        #     print(numpy_example)


        # The RLDSSpec for the RT1 dataset.
        rt1_spec = RLDSSpec(
            observation_info=b.info.features["steps"]["observation"],
            action_info=b.info.features["steps"]["action"],
        )

        trajectory_transform = TrajectoryTransformBuilder(
            rt1_spec, pattern_fn=n_step_pattern_builder(trajectory_length)
        ).build(validate_expected_tensor_spec=False)

        trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(ds)
        #pdb.set_trace()
        trajectory_datasets.append(trajectory_dataset)

    trajectory_dataset = tf.data.Dataset.sample_from_datasets(trajectory_datasets)
    trajectory_dataset = trajectory_dataset.map(
        get_observation_and_action_from_step, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle, batch, prefetch, repeat
    trajectory_dataset = trajectory_dataset.shuffle(batch_size * 16)
    trajectory_dataset = trajectory_dataset.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    trajectory_dataset = trajectory_dataset.repeat(num_epochs)
    trajectory_dataset = trajectory_dataset.prefetch(tf.data.AUTOTUNE)
    # pdb.set_trace()
    return iter(trajectory_dataset.as_numpy_iterator())


if __name__ == "__main__":
    #pdb.set_trace()
    ds = create_dataset(datasets=["fractal20220817_data"], split="train[:10]")
    it = next(ds)

    def print_shape(x):
        if isinstance(x, dict):
            shapes = tree.map_structure(lambda x: x.shape, x)
        else:
            shapes = x.shape
        return shapes

    shapes = tree.map_structure(print_shape, it)
    print(shapes)
