export CUDA_VISIBLE_DEVICES=1
dataset=EODP
model_name=MSF_GCN
train_epochs=20
patience=3

seq_len=96
label_len=0
learning_rate=0.001

down_sampling_layers=2 
e_layers=5       
d_model=16         
d_ff=32           
conv_channel=16    
node_dim=10        
gcn_depth=2
propalpha=0.3       
propbeta=0.3     

for pred_len in 16 48 96
do

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/EODP/ \
    --data_path data.csv \
    --adj_data adj_mx.pkl \
    --model_id $dataset \
    --model $model_name \
    --data PVA \
    --features M \
    --freq h \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers 1 \
    --enc_in 11 \
    --dec_in 11 \
    --c_out 11 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --down_sampling_layers $down_sampling_layers \
    --conv_channel $conv_channel \
    --propalpha $propalpha \
    --propbeta $propbeta \
    --dropout 0.1 \
    --gcn_depth $gcn_depth \
    --node_dim $node_dim \
    --itr 1  \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience 
done


