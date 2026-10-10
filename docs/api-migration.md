# Migrating from `ml_stack` to `poolhouse`

The package and the project were renamed: `ml_stack` is now `poolhouse` (`import poolhouse as ph`), the console
scripts are `poolhouse-*`, the project lives at github.com/adammikulis/poolhouse. There is no alias for the old
name; update the imports. The public names are listed in [api.md](api.md).

## The serving client and model discovery

| Old import | New |
|---|---|
| `from ml_stack import hub` | `from poolhouse import hub` (or `ph.hub`); the public names are `discover`, `ModelInfo`, `fetch`, `located`, `hub_cache` |
| `from ml_stack.client import Client` | `ph.client.Client` (`from poolhouse.client import Client`) |
| `from ml_stack.client import Request` | `ph.client.Request` |
| `from ml_stack.client import is_healthy` | `ph.client.is_healthy` |
| `from ml_stack.http import ServerError` | `ph.client.ServerError` (the same class as `poolhouse.http.ServerError`) |
| `from ml_stack.http import ServerUnreachable` | `ph.client.ServerUnreachable` |

`from poolhouse.client import Client` and `ph.client.Client` are the same object, so either spelling can be used
while a codebase is moved over. Code that is moved to the new name can use `ph.Error` to catch every failure of
the public API at once (`ServerError` is a different family: it is the failure of an HTTP request to a model).

## Not public

These are internal modules that were imported across the repository boundary; they are renamed the same way and
carry no promise, so expect them to change in any release. Ask for a public name rather than depending on them.

| Old import | New (internal) | Public replacement |
|---|---|---|
| `ml_stack.serve.broker_wire.lease` | `poolhouse.serve.broker_wire.lease` | `ph.serve.up(model)` leases a server and `ph.serve.down` releases it; for a lease held by the calling process, `ph.leases` is the table |
| `ml_stack.redteam.targets` | `poolhouse.redteam.targets` | none (the red-team harness is a tool, run as `python -m poolhouse.redteam`) |
| `ml_stack.net`, `ml_stack.net.scan`, `ml_stack.httpguard.Limits` | `poolhouse.net`, `poolhouse.net.scan`, `poolhouse.httpguard.Limits` | none |
