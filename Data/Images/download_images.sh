#!/bin/sh

aws s3 cp s3-eu-west-1.amazonaws.com/signality.football.background.play1.360p/frames_360p.zip .
unzip -o -j -qq frames_360.zip
rm frames_360.zip

