# Genkidama

Genkidama is a library offering a system for optimizing and distributing the computational workload required for training and tuning a machine learning algorithm over a local area network (LAN). In its first version it is conceived to be centered around Hyperparameter optimization using Gaussian Processes (by building a surrogate function estimating the cost function J(p_1, ... p_n), and then using an acquisition function to get the most appealing candidate points in the parameter space to be evaluated).

It does so acknowledging the fact that in most company there is an extensive need for CPU by the technical department, and lots of unused CPU by the other departments' machines.

By using the concept of Orchestrators and Computational nodes, it is possible to optimize the training of different algorithms cutting (linearly with respect to the number of computational nodes available) the computational time required for training and fine tuning of supervised algorithms.

More technically, the Orchestrator is built as an AsyncronousNode using DjangoChannels, working both as an optimizer and a workload balancer. It sends out the tasks through a secure Websocket connection to the computational nodes (which are SyncronousNodes check first the trustworthyness of the Orchestrator after accessing a secure database) and then silently and actively listens for directions (run in background) on what to train and what to tune. The process is brieflu describes as follows:

1. At inizialization of the Computational nodes, an instance in DB is written, containing the LAN IP of the computational node, and some variables containing information about the state of the node itself.

2. The controller tells the Orchestrator node to access the DB looking for active GenkiSocket nodes and saves them up, call the number of active nodes N.

3. Orchestrator gets the N samples N points uniformly in the parameter space and hands them out to the sockets (to evaluate the cost J(P), P vector of param values)

4. The orchestrator repeats the process iteratively, receiving back the evaluations of the cost from the nodes and updating real time the GP acquisition function, and handing out the new potentially most interesting points. This process is repeated in an asyncronous fashion until convergence or until a stopping condition is met.

5. The nodes recorded their status as "active" and leave that status until the data scientist stops the process, or when a node cannot receive requests as a GenkiSocket anymore. Then they're market as "inactive" and they die.



## Contributing
Pull requests and other contributors bringing in ideas and suggestions are warmly welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
