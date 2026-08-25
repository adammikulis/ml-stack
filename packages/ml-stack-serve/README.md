# ml-stack-serve

Starts a local model server, adopts one that is already running, and tears
it down again. Leases a server per model so two callers share one process.

## Install

```
pip install ml-stack-serve
```

Part of [ml-stack](https://github.com/adammikulis/ml-stack): train and run models across every machine on
your network.
