export CUDA_VISIBLE_DEVICES=3 
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id SMD \
  --model model \
  --data SMD \
  --features M \
  --seq_len 100 \
  --d_model 512 \
  --d_ff 512 \
  --ffn_dim 512 \
  --gpt_layer 6 \
  --anomaly_ratio 0.1 \
  --batch_size 4 \
  --patch_size 1 \
  --stride 1 \
  --top_k 10 \
  --train_epochs 3 \
  --learning_rate 1e-4 \
  --continue_training 0 \
  --n_head 8 \
  --num_layers 3 \
  --use_multi_gpu \
  --devices '0' \

  

  