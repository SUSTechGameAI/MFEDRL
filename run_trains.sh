#python train.py designer --n_envs 50 --rfunc_name "fun_c1" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_c1" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_b1" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_b1" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_cb1" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_bc1" --device "cuda:0";
#
#python train.py designer --n_envs 50 --rfunc_name "fun_c2" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_c2" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_b2" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_b2" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_cb2" --total_steps 500000  \
#--res_path "edrl-unigan/hist7/fun_bc2" --device "cuda:0";


#python train.py designer --n_envs 50 --rfunc_name "fun_c1" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_c1" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_b1" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_b1" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_cb1" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_bc1" --device "cuda:0";
#
#python train.py designer --n_envs 50 --rfunc_name "fun_c2" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_c2" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_b2" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_b2" --device "cuda:0";
#python train.py designer --n_envs 50 --rfunc_name "fun_cb2" --total_steps 10000  \
#--res_path "edrl-unigan/hist7/fun_bc2" --device "cuda:0";

python train.py designer_agsac --n_envs 50 --rfunc_name "fun_cb" --total_steps 1000000  \
--res_path "agsac/both" --device "cuda:0";
python train.py designer_agsac --n_envs 50 --rfunc_name "fun_c" --total_steps 1000000  \
--res_path "agsac/content" --device "cuda:0";python train.py designer_agsac --n_envs 5 --rfunc_name "fun_cb" --total_steps 100000 --res_path "agsac/both" --device "cuda:0"
python train.py designer_agsac --n_envs 50 --rfunc_name "fun_b" --total_steps 1000000  \
--res_path "agsac/behaviour" --device "cuda:0";

python train.py designer_agsac --n_envs 2 --rfunc_name "fun_cb" --total_steps 100000 --res_path "agsac/both" --device "cuda:0"