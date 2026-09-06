"""``ml-stack-surface``: every command's help and output, captured to a directory.

`ops` walks a checkout for its commands and subcommands, writes what each answers, and
compares two such directories. `probe` is the worker one command is asked through, run by
path so the ``ml_stack`` it reads is the checkout being asked. `cli` parses and prints.
"""
