# Credentials

Every token and key ml-stack reads comes through one resolver, `ml_stack.credentials`.
Nothing else in the library reads `HF_TOKEN` or an API key from the environment itself.

## Where a credential is looked for

`credentials.get(name)` returns the first of these that holds a value:

1. the `explicit` argument, when the caller passes one
2. the environment variable `NAME`
3. the file named by the environment variable `NAME_FILE` (the Docker and Kubernetes
   secrets convention; a mounted secret is usually root-owned and world-readable, so a file
   named this way is checked only for being a regular file under 64 KiB)
4. the credentials file, `~/.ml-stack/credentials.toml`, or the file named by
   `ML_STACK_CREDENTIALS_FILE` (`ML_STACK_HOME` moves the default)
5. for `HF_TOKEN` (and its older name `HUGGING_FACE_HUB_TOKEN`), Hugging Face's own token
   file: `HF_TOKEN_PATH`, else `$HF_HOME/token`, else `~/.cache/huggingface/token`. A token
   saved by `huggingface-cli login` is found without copying it.
6. the values stored with `--keychain`: wrapped under a subkey of the keystore master key
   (`docs/keystore.md`), kept in `~/.ml-stack/keystore/credentials.json`. A lookup that finds
   nothing there reads no keystore item; an item an older version stored one per name is moved
   in on first lookup and deleted once the wrapped copy reads back
7. nothing: `get` returns `None`, or raises `CredentialError` with `required=True`

A source that exists but cannot be trusted raises `CredentialError` instead of being skipped.
Silently falling through to a weaker source would hide the problem.

## The credentials file

```toml
HF_TOKEN = "hf_..."
ANTHROPIC_API_KEY = "..."
```

Top-level string values only. It is read only when:

- it is a regular file, at most 64 KiB, valid UTF-8 and valid TOML
- it belongs to the current user and is not readable or writable by group or others (mode
  `0600`); otherwise the error says `chmod 600 PATH`. Setting `ML_STACK_ALLOW_INSECURE_CREDENTIALS=yes-read-it-anyway`
  reads it anyway; nothing else does.
- if it is a symlink, its target is inside the directory that holds the link

Hugging Face's own token file is read with the ownership and size checks but not the mode
check, because `huggingface-cli login` writes it with the account's umask (usually `0644`).
Mode `0600` is still what to want there. The file is only read: ml-stack never copies it into
its own credentials file, a log or a child's environment unless that child is a server that
will download weights.

A value must be one printable line of at most 8192 characters; leading and trailing
whitespace and newlines are stripped, anything else (an embedded newline, a control
character) is refused, since a value ends up in an HTTP header.

On Windows the mode and ownership checks do not apply. `set` asks `icacls` to drop inherited
permissions on the file it writes, and says nothing when that fails; the file stays as safe as
the profile directory that holds it.

## Commands

```
ml-stack credentials set NAME            # hidden prompt, or stdin when it is not a terminal
echo "$TOKEN" | ml-stack credentials set NAME --stdin
ml-stack credentials set NAME --keychain # wrapped under the keystore master instead of plain in the file
ml-stack credentials list [--json]       # name, whether set, which source; never a value
ml-stack credentials unset NAME [--keychain]
ml-stack credentials path
```

A value is never taken from a command-line argument, because arguments appear in `ps` and in
shell history. `set` writes the file atomically (a temporary file in the same directory,
mode `0600`, renamed over the old one) in a directory created `0700`; it refuses a directory
other users can write.

## For a program embedding ml-stack

```python
from ml_stack import credentials

token = credentials.get("HF_TOKEN")                       # Secret | None
key = credentials.get("ANTHROPIC_API_KEY", required=True) # raises CredentialError
credentials.set("ANTHROPIC_API_KEY", value)               # file; keychain=True for the keychain
credentials.unset("ANTHROPIC_API_KEY")
credentials.status("ANTHROPIC_API_KEY")                   # {"present": True, "source": "credentials file"}
credentials.describe()                                    # one row per known name, no values
```

`get` returns a `Secret`, a `str` that prints as `Secret(<redacted>)` in `repr` (so a
traceback, a log of a dict or a debugger shows nothing) and cannot be pickled. Use it as a
string where it is needed and nowhere else. No function here puts a value in an exception
message or a log record.

## What it does not do

- It cannot stop the caller from printing the string it was given.
- It does not encrypt the file. Mode `0600` keeps other accounts out; it does not keep out
  root, a backup that copies the file, or malware running as you. The keychain is the better
  place on a machine others can image.
- It does not scrub the process's own memory.
