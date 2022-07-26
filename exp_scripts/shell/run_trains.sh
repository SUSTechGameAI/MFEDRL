python train.py sac_designer --n_envs 25 --rfunc_name 'fun_c' --total_steps 1000000  \
--res_path 'fc' --device 'cuda:0' --check_points 50000;
python train.py sac_designer --n_envs 25 --rfunc_name 'fun_b' --total_steps 1000000  \
--res_path 'fb' --device 'cuda:0' --check_points 50000 &
python train.py sac_designer --n_envs 25 --rfunc_name 'fun_cb' --total_steps 1000000  \
--res_path 'fc_fb' --device 'cuda:0' --check_points 50000;
