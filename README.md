# PLM-NR

This pipeline is based on the horovod framework. We run experiments on 4 GPUs. You need to split the original
behaviors.tsv file in the MIND dataset into X parts (named as behaviors_x.tsv, X∈ {0, 1, 2, ..., X}) if you use X GPUs
to train.

Run the run.sh script to train and test.
