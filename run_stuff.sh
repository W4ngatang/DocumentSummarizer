#!/bin/bash
#
#SBATCH -t 3-12:00 # Runtime
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH -o runs/run.out
#SBATCH -e runs/run.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -gpuid 1 -save_every 25 -epochs 13 -num_layers 3 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/three_layer_p.hdf5 > runs/three_layer.out
th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -rnn_size 250 -gpuid 1 -save_every 25 -epochs 13 -num_layers 1 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/rnn250_p.hdf5 > runs/rnn250.out
th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -rnn_size 750 -gpuid 1 -save_every 25 -epochs 13 -num_layers 1 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/rnn750_p.hdf5 > runs/rnn750.out
th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -kernel_width 4 -gpuid 1 -save_every 25 -epochs 13 -num_layers 1 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/wkern4_p.hdf5 > runs/wkern4.out
th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -num_kernels 200 -gpuid 1 -save_every 25 -epochs 13 -num_layers 1 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/nkern200_p.hdf5 > runs/nkern200.out
th train.lua -data_file data/cnn-extract-train.hdf5 -val_data_file data/cnn-extract-val.hdf5 -num_kernels 300 -gpuid 1 -save_every 25 -epochs 13 -num_layers 1 -pre_word_vecs_enc data/embeds.hdf5 -predfile runs/nkern300_p.hdf5 > runs/nkern300.out
