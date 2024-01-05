#!/bin/bash

source activate ~/anaconda3/bin/conda.sh
conda activate deeppocket


python predict.py -p ./data_dir/protein.pdb -c ./classifier_models/first_model_fold1_best_test_auc_85001.pth.tar -s segmentation_models/seg0_best_test_IOU_91.pth.tar -r 3
