##train
python train.py  --name ME_LEDA   --model meunet   --dataroot /data/zhchen/Mayo2016_2d --lr 0.0001 --gpu_ids 7 --print_freq 25 --batch_size 1 --lr_policy cosine


##test
python test.py  --name ME_LEDA   --model meunet  --result_name me_results   --gpu_ids 6 --batch_size 1 --eval