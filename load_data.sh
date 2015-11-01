#!/bin/bash

languages="bg cs da de el es et fi fr hu it lt lv nl pl pt ro sk sl sv"

# create a folder for data storage
if [ ! -d data ]; then
    mkdir data
fi

for language in $languages
do
    echo "Downloading $language corpus ..."
    wget wget -qO- "http://www.statmt.org/europarl/v7/$language-en.tgz" | tar xvz -C "./data"
done

echo "Done downloading the dataset"

# create a folder for experiments
mkdir -p experiment

for f in data/*
do
    if [[ "$f" != *\.en ]]
    then
        head -10000 "$f" > "experiment/${f: -2}"

    elif [[ "$f" == *\.fr-en.en ]]
    then
        head -10000 "$f" > "experiment/${f: -2}"
    fi
done

echo "Experiment data created"