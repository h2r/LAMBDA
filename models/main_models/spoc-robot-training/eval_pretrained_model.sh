export OBJAVERSE_DATA_BASE_DIR="objaverse_assets"
export OBJAVERSE_HOUSES_BASE_DIR="objaverse_houses"
export OBJAVERSE_DATA_DIR="objaverse_assets/2023_07_28"
export OBJAVERSE_HOUSES_DIR="objaverse_houses/houses_2023_07_28"
export CKPT_DIR="pretrained_models"
export PYTHONPATH="./"

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate spoc

echo "Start evaluation"
python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir tmp_log \
 --test_augmentation --task_type ObjectNavType \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 nav_task_relevant_object_bbox manip_task_relevant_object_bbox nav_accurate_object_bbox manip_accurate_object_bbox \
 --house_set objaverse --wandb_logging False --num_workers 1 \
 --gpu_devices 0 1 --training_run_id SigLIP-ViTb-3-double-det-CHORES-S --local_checkpoint_dir $CKPT_DIR
