# MARL

This repo aims at providing a flexible system for distributed training of Reinforcement Learning algorithms, exploiting concepts such as Networking using websockets to connect and orchestrate multiple machines connected to a network, as well as multithreading and multiprocessing to optimize the use of resources within each of the working machines.

## Functioning
The algorithm underlying all of this is the following at the time being (it's simplified to give a general idea, there is a lot of coordination with semaphores, waiting and parameters under the hood):

1. Initialize N childs (working nodes) called {c_0, ..., c_(N-1)}.
2. Initialize S, a parent.
3. (Asynchronous) For c_i in c_0, ..., c_(N-1):
   1. S -> c_i with a handshake.
   2. c_i -> S ackowledges handshake and agree on weights bytes length.
   3. c_i initializes the same network structure and a shared replay buffer amongst its cpu cores, share the optimizer connected to the NN with all cpus.
   3. S -> c_i sends current network parameters, c_i starts working.
   4. (Parallel) For c_i_cpucore_j in c_i_cpucore_1, ..., c_i_cpucore_M:
       1. c_i_cpucore_0 is the designated cpu for contact with S, it's not going to be working.
       2. If c_i_cpucore_j != c_i_cpucore_0:
          1. Initialize a core-specific neural net.
          2. For episode in range(len(episodes)):
             1. While episode is not done:
                1. Play k steps and save results (s, a, r) in the shared buffer
                2. Random sample batch and update shared optimizer, acting on the c_i NN weights.
                3. If done: break loop
       3. if c_i_cpucore_0 and all cpus have finished episode: Send weights to S
       4. if c_i_cpucore_0 and all cpus have finished episode: Wait for the new weights to come in from S, update the current c_i ones.
       5. c_i_cpucore_j updates its parameter with the current c_i ones

### Note: In any case the main implementation of this is done in the file SingleCore.py, where each single parameter is thoroughly explained

## Installation

```bash
pip3 install requirements.txt
```

## Usage of the classic A3C algorithm
Initialize a set of parameters inside the file a3c_single_machine.py, then to start it use the following command (note that this version does not yet make use of arguments in the shell call).
```
python3 a3c_single_machine.py
```
It is going to produce an output of weights (.pt) which can then be fed to test_env.py to visualize the un-normalized distribution of the rewards gotten by the algorithm in the wild.

## Usage of the distributed 'A4C' algorithm (name to be decided)
Install in the different machines this same package, and define the parameters both in init_parent.py and init_child.py. It is important that at this point to have the LAN address of each of the child (worker) nodes (parent node is not necessary). Note: in this version the active port must be the same for every node. Activate then the wss servers on each of the child node with the following command
```
python3 init_child.py -ip <lan_ip> -p <port_n>
```
After all of the child servers are up and running, we're going to have to initialize the client (the parent node) with the following:
```
python3 init_parent.py -ip <child_ip_0> <child_ip_1> ... <child_ip_n> -p <port_n>
```

## Steps
- [x] Implement multiprocessing
- [x] Implement websockets protocol
- [x] Implement multithreading on the parent
- [x] Test successfully learning of a3c
- [x] Test successfully learning of a4c
- [x] Implement with other RL algorithms (perhaps a custom Environment)
- [ ] Implement replay memory buffer shared amongst CPU for R.S.
- [ ] Open to other networks outside LAN (use security protocols and hashes)
- [ ] Try to do the same with Quantization (reducing bytes size of param, from f32 to less, makes the transfer of params faster)

## License
[MIT](https://choosealicense.com/licenses/mit/)
