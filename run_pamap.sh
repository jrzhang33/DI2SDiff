# PAMAP2 using 20% training data 
# Target 0
python Style_conditioner/main_train_style.py --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 0 --training_mode 'self_supervised' --logs_save_dir './Style_conditioner/conditioner_pth/'
python Diffusion_model/main_train_diff.py  --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 0 --results_folder './Diffusion_model/dm_pth/'
python Featurenet/main_train.py --seed 1 --dataset 'pamap' --remain_rate 0.2 --target 0 --batch_size 64 --step1 0 --Ocomb 5 --Ktimes 2 --lr_decay_cls_f 1 --lr_decay_cls 1 --lr_decay_ori 1 --lr_decay_ori_f 1e-1 --lr_decay1 1e-2 --lr_decay2 1 --lr 1e-3

# Target 1
python Style_conditioner/main_train_style.py --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 1 --training_mode 'self_supervised' --logs_save_dir './Style_conditioner/conditioner_pth/'
python Diffusion_model/main_train_diff.py  --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 1 --results_folder './Diffusion_model/dm_pth/'
python Featurenet/main_train.py --seed 1 --dataset 'pamap' --remain_rate 0.2 --target 1 --batch_size 64 --step1 0 --Ocomb 5 --Ktimes 2 --lr_decay_cls_f 1 --lr_decay_cls 1 --lr_decay_ori 1 --lr_decay_ori_f 1e-1 --lr_decay1 1e-2 --lr_decay2 1e-5 --lr 7e-3

# Target 2
python Style_conditioner/main_train_style.py --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 2 --training_mode 'self_supervised' --logs_save_dir './Style_conditioner/conditioner_pth/'
python Diffusion_model/main_train_diff.py  --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 2 --results_folder './Diffusion_model/dm_pth/'
python Featurenet/main_train.py --seed 1 --dataset 'pamap' --remain_rate 0.2 --target 2 --batch_size 64 --step1 0 --Ocomb 2 --Ktimes 1 --lr_decay_cls_f 1 --lr_decay_cls 1 --lr_decay_ori 1 --lr_decay_ori_f 1e-1 --lr_decay1 1e-2 --lr_decay2 1 --lr 8e-3

# Target 3
python Style_conditioner/main_train_style.py --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 3 --training_mode 'self_supervised' --logs_save_dir './Style_conditioner/conditioner_pth/'
python Diffusion_model/main_train_diff.py  --seed 1 --selected_dataset 'pamap' --remain_rate 0.2 --target 3 --results_folder './Diffusion_model/dm_pth/'
python Featurenet/main_train.py --seed 1 --dataset 'pamap' --remain_rate 0.2 --target 3 --batch_size 64 --step1 0 --step2 0 --Ocomb 10 --Ktimes 1 --lr_decay_cls_f 1 --lr_decay_cls 1  --lr 9e-3


