num_clients=8
server_port=50000
server_ip=127.0.0.1
local_epochs=2
active_mode=adptpolu
dataset=PeMSD4SPEED
mode=FED
num_nodes=307
dsp=32
dsu=32
device='cuda:3'
declare -A rand_nums
rand_nums[$server_port]=1

echo "server ip: $server_ip"
echo "server port: $server_port"

python server.py -n $num_clients -p $server_port -i $server_ip -N $num_nodes -dsp $dsp -dsu $dsu --device $device &

(sleep 0.01

for i in $(seq 1 $num_clients)
do
    sleep 0.1
    client_port=$((RANDOM % 40001 + 20000))
    while [[ -n ${rand_nums[$client_port]} ]]; do
        client_port=$((RANDOM % 40001 + 20000))
    done
    rand_nums[$client_port]=1

    echo "client $i port: $client_port"
    python client.py $dataset $mode --cid $i -sip $server_ip -sp $server_port -cp $client_port --device $device --num_clients $num_clients --divide metis --fedavg --active_mode $active_mode --act_k 2 --local_epochs $local_epochs &
done
)
