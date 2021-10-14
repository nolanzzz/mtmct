cd src
python train.py mot --exp_id crowdhuman_dla34 --gpus 0,1 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..


# run tracker on train dataset first (cam 4,5 left)
# then run tracker on test dataset (cam 3,4,5 left)