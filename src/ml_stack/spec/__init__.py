"""Tree speculative decoding on MLX for hybrid DeltaNet/attention models.

A drafter proposes a tree of tokens; `TreeVerifier` scores every node in one pass of the
target; an acceptance rule keeps a path; the caches are advanced along it. `Engine` holds a
loaded target, its drafter and the session that reuses a prompt's prefix across calls.
"""
