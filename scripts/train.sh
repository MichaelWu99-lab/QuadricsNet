#!/bin/bash

# points
python -u train_01.py configs/configs_train/config_quadrics.yml sphere 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics.yml plane 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics.yml cylinder 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics.yml cone 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001

# points + normals
python -u train_01.py configs/configs_train/config_quadrics_normal.yml sphere 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics_normal.yml plane 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics_normal.yml cylinder 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
python -u train_01.py configs/configs_train/config_quadrics_normal.yml cone 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001