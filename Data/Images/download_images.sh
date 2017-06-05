#!/bin/sh

#wget https://s3-eu-west-1.amazonaws.com/signality.football.background.play1.360p/frames_360p.zip
#unzip -o -qq frames_360p.zip
#rm frames_360p.zip

mkdir Data

aws s3 sync s3://signality.football.line-annotated-games Data/.

mkdir Data/Images

echo "{}" > Data/annotations.json
for d in Data/signality.*/; do
    mv $d/*.png Data/Images/.

    jq -s '.[0] * .[1]' Data/annotations.json $d/annotations.json > Data/tmp.json
    mv Data/tmp.json Data/annotations.json

    rm -rf $d
done
