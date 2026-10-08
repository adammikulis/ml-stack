import { h } from "./base.js";

export function messageRoute(board) {
  const view = board.view;
  if (view.kind === "board" && board.channelTab === "messages") return ["messages", {board:view.name}];
  if (view.kind === "thread") return ["thread", {root:view.root}];
  if (view.kind === "dm") return ["dm", {a:view.a, b:view.b}];
  return null;
}

export async function loadPage(board, direction = "latest") {
  const route = messageRoute(board);
  if (!route) return;
  const revision = board.viewRevision;
  const pageRevision = board.pageRevision = (board.pageRevision || 0) + 1;
  const cursor = direction === "older" ? {before:board.items[0]?.seq || 0}
    : direction === "newer" ? {after:board.items.at(-1)?.seq || 0} : {};
  const answer = await board.get(route[0], {...route[1], ...cursor});
  if (revision !== board.viewRevision || pageRevision !== board.pageRevision || board.stopped) return;
  const incoming = answer.messages || [];
  const combined = direction === "older" ? [...incoming, ...board.items]
    : direction === "newer" ? [...board.items, ...incoming] : incoming;
  const unique = [...new Map(combined.map(message => [message.seq, message])).values()];
  board.items = direction === "older" ? unique.slice(0, 200) : unique.slice(-200);
  board.page = {...answer, has_older:answer.has_older || direction === "newer" && (board.page?.has_older || unique.length > 200),
    has_newer:answer.has_newer || direction === "older" && board.page?.has_newer || unique.length > 200 && direction === "older"};
}

export function pageControls(board) {
  if (!messageRoute(board)) return null;
  const action = (label, direction, disabled) => h("button", {type:"button", disabled,
    onclick:async () => {
      try { await loadPage(board, direction); board.error = ""; }
      catch (error) { if (error.name !== "AbortError") board.error = String(error.message).slice(0, 160); }
      board.update();
    }}, label);
  return h("div", {class:"history-controls"},
    action("Load older messages", "older", !board.page?.has_older),
    action("Load newer messages", "newer", !board.page?.has_newer),
    action("Jump to latest", "latest", false));
}

export function renderFeed(board, nodes) {
  const anchor = [...board.feed.querySelectorAll("[data-seq]")].find(node => node.offsetTop >= board.feed.scrollTop);
  const offset = anchor ? anchor.offsetTop - board.feed.scrollTop : 0;
  const scroll = board.feed.scrollTop;
  const old = new Map([...board.feed.querySelectorAll("[data-seq]")].map(node => [node.dataset.seq, node]));
  const reusable = nodes.map(node => {
    const previous = old.get(node?.dataset?.seq);
    return previous && previous.textContent === node.textContent ? previous : node;
  });
  board.feed.replaceChildren(...reusable);
  const retained = anchor && board.feed.querySelector(`[data-seq="${anchor.dataset.seq}"]`);
  board.feed.scrollTop = retained ? retained.offsetTop - offset : scroll;
}
