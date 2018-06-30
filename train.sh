for i in 16 32 64
do
for k in 128 256 512 1024 2048 4096
do
 echo "bits:"$i"   dim:"$k
 CUDA_VISIBLE_DEVICES=1 python train.py $k 1000 $k
done
done
