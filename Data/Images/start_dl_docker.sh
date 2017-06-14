#! /bin/sh

nvidia-docker run --rm -e DISPLAY=$DISPLAY -ti \
   -v ~/:/home/saif \
   --net=host \
   dl:1.1 bash
