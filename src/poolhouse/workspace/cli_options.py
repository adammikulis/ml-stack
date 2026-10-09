"""The options the `poolhouse-workspace` commands share."""

from __future__ import annotations

from poolhouse.command import flag, option
from poolhouse.workspace import tokens
from poolhouse.workspace.identity import TOKEN_ENV

__all__ = ["COMMON", "FOR_AGENT", "NODE_COMMON", "READ", "WIDEN"]

COMMON = [option("json"),
          flag("--request-id", default="", help="reuse an exact remote mutation request after a lost response"),
          flag("--token-file", default="", help=f"a file holding the sender's token "
                                                f"(else --agent, else ${TOKEN_ENV}, "
                                                f"else ${tokens.AGENT_ENV})"),
          flag("--agent", default="", help="which of your own token files to use, by agent name (else "
                                           f"${tokens.AGENT_ENV}); the token, not this name, decides who is sending")]
NODE_COMMON = [option("json"),
               flag("--token-file", default="", help=f"a file holding the sender's token (else --agent, else "
                                                     f"${TOKEN_ENV}, else ${tokens.AGENT_ENV})"),
               flag("--agent", default="", help="which of your own tokens to use, by name (else "
                                                f"${tokens.AGENT_ENV}); the token, not this name, decides who is sending"),
               flag("--board", default="", help="the board to use (default: the project this directory belongs to)")]
WIDEN = [flag("--limit", type=int, default=0,
              help="show this many (default: a few, each cut short; the rest is counted)"),
         flag("--all", action="store_true", help="show everything, uncut")]
READ = [*WIDEN, flag("--ack", action="store_true", help="mark what is shown as read"),
        flag("--raw", action="store_true", help="also show the unfenced text of clear messages")]
FOR_AGENT = flag("--for-agent", default="", metavar="NAME",
                 help="show only this agent's records, by its unique name or an unambiguous start of it; "
                      "it selects what to read and never sets who is sending (for scratch folders, which "
                      "are private, only a lead or person may name another agent)")
