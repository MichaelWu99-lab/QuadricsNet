#!/bin/bash
source activate QuadricsNet
mat

#
python -u train_01.py configs/configs_train/config_quadrics_normal.yml cylinder 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001
# python -u train_01.py configs/configs_train/config_quadrics_normal.yml cone 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001
# python -u train_01.py configs/configs_train/config_quadrics_normal.yml sphere 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001


#
# python -u train_01.py configs/configs_train/config_quadrics_syth_normal.yml elliptic_cone 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001
# python -u train_01.py configs/configs_train/config_quadrics_syth.yml elliptic_cone 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001

# python -u train_01.py configs/configs_train/config_quadrics_syth.yml elliptic_cylinder 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001
# python -u train_01.py configs/configs_train/config_quadrics_syth.yml ellipsoid 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001
# python -u train_01.py configs/configs_train/config_quadrics_syth.yml elliptic_cylinder 0.2 0.2 0.0 0.0 0.0 0.2 0.2 0.001



# python -u train_01.py configs/configs_train/config_quadrics_syth.yml ellipsoid 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001

# python -u train_01.py configs/configs_train/config_quadrics_normal.yml cylinder 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001

# nohup bash scripts/train_quadrics.sh > train_log/train_quadrics_1_elliptic_cone_0.log 2>&1 &

# nohup python -u train_01.py configs/configs_train/config_quadrics_syth.yml elliptic_cylinder 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.001 > train_log/train_quadrics_0_elliptic_cylinder.log 2>&1 &

# rsync -avz --delete -e "ssh -i /root/.ssh/id_rsa -p 20120" --include={'test_anything/','test_anything/test_anything_OneShape.py'} --exclude={'data.7z','.git*','logs/results*','test_anything/data_to_fit/*','test_anything/output/*','.vscode/'}  /root/wuji/QuadricsNet/ root@10.254.1.155:/root/wuji/QuadricsNet_syn/

# rsync -avz --delete -e "ssh -i /root/.ssh/id_rsa -p 20148" --include={'test_anything/'} --exclude={'data.7z','.git*','logs/results*','test_anything/data_to_fit/*','test_anything/output/*','.vscode/'}  /root/wuji/QuadricsNet/ root@10.254.1.155:/root/wuji/QuadricsNet_syn/