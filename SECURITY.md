# Security policy

## Supported versions

ml-stack is pre-1.0. Only the latest release receives security fixes.

## Reporting a vulnerability

Report privately through GitHub private vulnerability reporting:
**Security tab > Report a vulnerability** (https://github.com/adammikulis/ml-stack/security/advisories/new).
Please do not open a public issue or pull request for a vulnerability.

Include the version, the machine type, what you did, what happened and what you expected. A proof of concept helps.

Expect an acknowledgement within 7 days and a status update within 30 days. This is a one-person project and
timelines are best effort. Reporters are credited in the advisory unless they ask not to be.

## Scope

In scope: the peer daemon (`ml-stack-traind`) and its HTTP API, peer discovery and the passphrase-derived keys,
the job runner and the commands it launches, the model server wrapper, the MCP server (`ml-stack-mcp`), the graph
page server (`ml-stack-graph serve`), the installers in `packaging/` and the release workflows.

Out of scope: vulnerabilities in llama.cpp, PyTorch, MLX, Tauri, PyInstaller or any other dependency (report them
upstream); anything that needs an attacker who already has a shell on a cluster machine; hostile peers that hold
the cluster passphrase.

## Deployment notes

- Peers talk plain HTTP on the local network and authenticate with a key derived from the shared passphrase. Use ml-stack on a
  network you trust. Do not expose the daemon port to the internet.
- Choose a long passphrase: anyone who learns it can send work to every machine in the cluster, and the work is
  commands that run as the user that started the daemon.
- The one-line installers download from the `main` branch. To pin a version, fetch the installer from a release tag
  (`.../ml-stack/<tag>/packaging/install.sh`) and read it before running it.
- Release assets are built in GitHub Actions; see `docs/release.md` for how to verify them.
