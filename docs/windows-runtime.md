# Windows runtime

The Windows `ml-stack` launcher runs the daemon and model backend in Ubuntu on WSL and opens
the interface in your Windows browser at `http://127.0.0.1:8770/ui/`.
Ubuntu must use the virtual-machine WSL architecture. The launcher checks its kernel,
Python interpreter, and bubblewrap namespaces before starting the application.

Install Ubuntu through Windows' WSL setup, then install these packages in Ubuntu:

```sh
sudo apt install python3-venv bubblewrap clamav
sudo freshclam
```

Install the NVIDIA driver on Windows to expose the GPU to Ubuntu. The launcher does not
change administrator settings, firewall rules, or WSL configuration. Set
`ML_STACK_WSL_DISTRO` to select an Ubuntu distribution with a different name.

Run `ml-stack` from the local source installation on Windows. Its first launch creates a
Python environment under `~/.local/share/ml-stack/runtime` in Ubuntu and installs the
application's dependencies. Subsequent launches reuse it; changes to the source location
or dependency declaration refresh the installation. Models download only when you choose
to install one in the interface. State and caches live in Ubuntu by default. Windows
`ML_STACK_HOME`, `ML_STACK_CACHE`, and explicit daemon filesystem arguments are translated
to Linux paths.

The model process runs in bubblewrap with its own network namespace. It communicates
through a private Unix socket to a loopback relay owned by its server supervisor. The GPU
device and WSL driver libraries are mounted explicitly. The broker owns the supervisor's
lease and process lifetime.

The Linux model confinement tests cover granted reads, denied reads outside the grant,
denied access to host loopback, GPU discovery, and streamed relay replies. Model inference
requires a CUDA-capable managed server and downloaded weights. GPU discovery alone does
not verify inference or the MTP head. Native Windows model sandboxing is not implemented.
The generic Linux sandbox's network-sharing modes and executable allow-list need additional
enforcement; model serving uses the network-denied mode.
