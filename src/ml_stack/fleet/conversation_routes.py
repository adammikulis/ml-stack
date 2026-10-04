"""Saved conversation listing, lookup and updates behind the daemon UI guard."""
from __future__ import annotations


class ConversationRoutes:
    def route(self) -> bool:
        if self.path.startswith("/ui/conversations"):
            try:
                return self._conversations()
            except (TypeError, ValueError) as error:
                self.send(400, {"error": str(error)})
                return True
        return super().route()

    def _body(self) -> dict:
        request = self.body()
        if not isinstance(request, dict):
            raise ValueError("conversation request must be an object")
        if set(request) - {"model", "title", "settings"}:
            raise ValueError("unknown conversation field")
        for field in ("model", "title"):
            if field in request and not isinstance(request[field], str):
                raise ValueError(f"{field} must be text")
        if "settings" in request and not isinstance(request["settings"], dict):
            raise ValueError("conversation settings must be an object")
        return request

    def _conversations(self) -> bool:
        store = self.ui.conversations
        if store is None:
            self.send(501, {"error": "no chat store on this daemon"})
            return True
        cid = self.path[len("/ui/conversations"):].strip("/")
        if cid:
            return self._one_conversation(cid)
        if self.method == "GET":
            self.send(200, {"conversations": [conversation.public(full=False)
                                             for conversation in store.search(self.asked("q"))]})
            return True
        if self.method == "POST":
            request = self._body()
            if not isinstance(request.get("model", ""), str) or not isinstance(request.get("title", ""), str):
                raise ValueError("model and title must be text")
            made = store.start(model=request.get("model", ""), title=request.get("title", ""),
                               settings=request.get("settings"))
            self.send(201, made.public())
            return True
        return super().route()

    def _one_conversation(self, cid: str) -> bool:
        store = self.ui.conversations
        found = store.get(cid)
        if found is None:
            self.send(404, {"error": "no such chat"})
            return True
        if self.method == "GET":
            self.send(200, found.public())
            return True
        if self.method == "DELETE":
            store.remove(cid)
            self.send(200, {"removed": cid})
            return True
        if self.method == "POST":
            request = self._body()
            changed = store.update(cid, model=request.get("model"), title=request.get("title"),
                                   settings=request.get("settings"))
            if changed is None:
                self.send(404, {"error": "no such chat"})
            else:
                self.send(200, changed.public())
            return True
        return super().route()
