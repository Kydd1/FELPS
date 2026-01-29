# fcfs
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting llama-7b & 
server_pid=$!
(
  sleep 45
  python run_exp.py --model-setting llama-7b --mode real --suite multitask --trace-dir multitask --output multitask/fcfs
  sleep 20
  kill $server_pid
  pkill -f "slora"
)
# felps 
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting llama-7b --scheduler felps --lambda-value 10 --ema 100 &
server_pid=$!
(
  sleep 45  
  python run_exp.py --model-setting llama-7b --mode real --suite multitask --trace-dir multitask --output multitask/felps-10-100
  sleep 20
  kill $server_pid
  pkill -f "slora"
)

sleep 5
wait
# vtc
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting llama-7b --scheduler vtc_fair &
server_pid=$!
(
  sleep 45
  python run_exp.py --model-setting llama-7b --mode real --suite multitask --trace-dir multitask --output multitask/vtc
  sleep 20
  kill $server_pid
  pkill -f "slora"
)
sleep 5
wait
# ac 5
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting llama-7b --batch-num-adapters 5 &
server_pid=$!
(
  sleep 45
  python run_exp.py --model-setting llama-7b --mode real --suite multitask --trace-dir multitask --output multitask/ac5 --append
  sleep 20
  kill $server_pid
  pkill -f "slora"
)
sleep 5
wait
wait
# inlaid
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting llama-7b --scheduler inlaid --batch-num-adapters 5 &
server_pid=$!
(
  sleep 45  
  python run_exp.py --model-setting llama-7b --mode real --suite multitask --trace-dir multitask --output multitask/inlaid
  sleep 20
  kill $server_pid
  pkill -f "slora"
)
sleep 5
wait