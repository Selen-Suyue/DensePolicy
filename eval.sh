eval_args=(
    --ckpt logs/pour_2/policy_last.ckpt
    --calib /home/ubuntu/data/calib/1734524187764
    --Tp 16 --Ta 16 --voxel_size 0.005
    --obs_feature_dim 512 --hidden_dim 512
    --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1
    --dim_feedforward 2048 --dropout 0.1
    --max_steps 300 --seed 233
    --discretize_rotation --ensemble_mode act
    #--vis
    --video_save_filedir "your video save path"
)

python eval.py "${eval_args[@]}"