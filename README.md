# Genkidama

Genkidama is a library offering a system for optimizing and distributing the computational workload required for training and tuning a machine learning algorithm over a local area network (LAN). 

It does so acknowledging the fact that in most company there is an extensive need for CPU by the technical department, and lots of unused CPU by the other departments' machines.

By using the concept of Orchestrators and Computational nodes, it is possible to optimize the training of different algorithms cutting (linearly with respect to the number of computational nodes available) the computational time required for training and fine tuning of supervised algorithms.

More technically, the Orchestrator is built as an AsyncronousNode using DjangoChannels, working both as an optimizer and a workload balancer. It sends out the tasks through a secure Websocket connection to the computational nodes (which are SyncronousNodes check first the trustworthyness of the Orchestrator after accessing a secure database) and then silently and actively listens for directions (run in background) on what to train and what to tune. Computational nodes send then back the result to the Orchestrator as soon as they're done, and the process is repeated iteratively either until convergence or until a stopping condition is met.


1. The Data Scientist tells the Orchestrator node to scan the network looking for GenkiSocket nodes available (first by pinging the LAN IPs and then filtering them down by sending websocket requests to each of them to verify if they're actually GenkiSockets) then stores them in the database.

2. Given this list then the GenkiSockets are going to be marked as "active" and the workload of the algorithm can be distributed amongst them.

3. The orchestrator repeats the process iteratively until convergence or until a stopping condition is met.

4. The nodes record their status as "active" and leave that status until the data scientist stops the process, or when a node cannot receive requests as a GenkiSocket anymore. Then they're market as "inactive" and they die.



## Contributing
Pull requests and other contributors bringing in ideas and suggestions are warmly welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
