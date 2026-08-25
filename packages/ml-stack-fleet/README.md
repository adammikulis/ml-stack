# ml-stack-fleet

Finds the other machines on your network and runs work on them. Machines
that share a passphrase discover each other over multicast, sign what they
say to each other, and pass jobs, files and models between themselves.

It also carries the daemon and the interface that drives all of it.

## Install

```
pip install ml-stack-fleet
```

## Running it

```
ml-stack                 # start this machine and open the interface
ml-stack-peers setup     # join a cluster from a terminal
ml-stack-peers ls        # what else is on the network
```

Part of [ml-stack](https://github.com/adammikulis/ml-stack): train and run models across every machine on
your network.
