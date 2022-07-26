@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_b/20" --n 6
@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_b/30" --n 6
@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_c/015" --n 6
@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_c/020" --n 6
@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_bc/20_015" --n 6
@REM python gen_samples.py --src_path "exp_data/edrl-unigan/fun_bc/30_020" --n 6

python gen_samples.py --src_path "exp_data/edrl-unigan/fun_b/20" --n 30 --folder "tests"
python gen_samples.py --src_path "exp_data/edrl-unigan/fun_b/30" --n 30 --folder "tests"
python gen_samples.py --src_path "exp_data/edrl-unigan/fun_c/015" --n 30 --folder "tests"
python gen_samples.py --src_path "exp_data/edrl-unigan/fun_c/020" --n 30 --folder "tests"
python gen_samples.py --src_path "exp_data/edrl-unigan/fun_bc/20_015" --n 30 --folder "tests"
python gen_samples.py --src_path "exp_data/edrl-unigan/fun_bc/30_020" --n 30 --folder "tests"
