nvidia-docker run --rm \
-e DISPLAY=$DISPLAY \
-ti \
-v /home/ubuntu/disk1:/home/ubuntu \
--net=host \
-w /home/ubuntu/background_display \
dl:1.1 \
python3 -m homography.deep_homo_tf
