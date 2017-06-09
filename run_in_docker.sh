nvidia-docker run --rm \
-e DISPLAY=$DISPLAY \
-ti \
-v /home/ubuntu/disk1:/home/ubuntu \
--net=host \
-w /home/ubuntu/bg \
dl:1.1 \
python3 -m homo.deep_homo_tfss
