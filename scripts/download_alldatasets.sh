#! /bin/sh

[ ! -d "dataset" ] && mkdir dataset

for i in "PXB184" "RLW104" "TXB805" "GQS883"
do
    curl -L https://github.com/triphop/audio2photoreal_handson/releases/download/v0.1/${i}.tgz -o ${i}.tgz || { echo 'downloading dataset failed' ; exit 1; }
    tar xvzf ${i}.tgz -C dataset/
    rm ${i}.tgz
done
