@REM python simulate.py --path "exp_data/edrl-unigan/fun_b/20/samples" --n_parallel 6
@REM python simulate.py --path "exp_data/edrl-unigan/fun_b/30/samples" --n_parallel 6
@REM python simulate.py --path "exp_data/edrl-unigan/fun_c/015/samples" --n_parallel 6
@REM python simulate.py --path "exp_data/edrl-unigan/fun_c/020/samples" --n_parallel 6
@REM python simulate.py --path "exp_data/edrl-unigan/fun_bc/20_015/samples" --n_parallel 6
@REM python simulate.py --path "exp_data/edrl-unigan/fun_bc/30_020/samples" --n_parallel 6

python simulate_batched.py --path "exp_data/randlvls/l5" --n_parallel 4
python simulate_batched.py --path "exp_data/randlvls/l6" --n_parallel 4
python simulate_batched.py --path "exp_data/randlvls/l7" --n_parallel 4
python simulate_batched.py --path "exp_data/randlvls/l8" --n_parallel 4
