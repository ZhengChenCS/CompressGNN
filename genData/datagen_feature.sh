root=../../dataset

dataset=cnr-2000

if [ ! -d "${root}/${dataset}/origin/feature_scale/" ]; then
    mkdir -p ${root}/${dataset}/origin/feature_scale
    mkdir -p ${root}/${dataset}/origin/feature_scale/coo
    mkdir -p ${root}/${dataset}/origin/feature_scale/csr
fi

if [ ! -d "${root}/${dataset}/compress/feature_scale/" ]; then
    mkdir -p ${root}/${dataset}/compress/feature_scale
    mkdir -p ${root}/${dataset}/compress/feature_scale/coo
    mkdir -p ${root}/${dataset}/compress/feature_scale/csr
fi


feature_dim=(1 2 4 8 16 32 64 128 256 512 1024)

for i in {0..10}
do
    python datagen_feature.py \
    --data=${root}/${dataset}/origin/data_coo.pt \
    --scale_factor=${feature_dim[i]} \
    --output=${root}/${dataset}/origin/feature_scale/coo/${feature_dim[i]}.pt
done

for i in {0..10}
do
    python datagen_feature.py \
    --data=${root}/${dataset}/origin/data_csr.pt \
    --scale_factor=${feature_dim[i]} \
    --output=${root}/${dataset}/origin/feature_scale/csr/${feature_dim[i]}.pt
done

for i in {0..10}
do
    python datagen_feature.py \
    --data=${root}/${dataset}/compress/data_coo_1.pt \
    --scale_factor=${feature_dim[i]} \
    --output=${root}/${dataset}/compress/feature_scale/coo/${feature_dim[i]}.pt
done

for i in {0..10}
do
    python datagen_feature.py \
    --data=${root}/${dataset}/compress/data_csr_1.pt \
    --scale_factor=${feature_dim[i]} \
    --output=${root}/${dataset}/compress/feature_scale/csr/${feature_dim[i]}.pt
done
