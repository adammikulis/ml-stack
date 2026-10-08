"""Request-bound person authority for browser workspaces."""

from ml_stack.fleet.session import parse_cookie
from ml_stack.workspace import boardroute
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied, Identity
from ml_stack.workspace.service import Workspace


class SetupRequired(Denied):
    pass


class SessionWorkspace(Workspace):
    def __init__(self, host, request, project_id):
        self._host = host
        self._request = request
        self._project_id = project_id
        self._actor = object()
        self._session = request.ui.sessions.get(parse_cookie(request.cookie))
        super().__init__(host.projects.workspace_base(project_id))
        self._validate()

    def _validate(self):
        request = self._request
        ui = request.ui
        if (ui.workspaces is not self._host or ui.projects is not self._host.projects
                or self._session is None
                or ui.sessions.get(parse_cookie(request.cookie)) is not self._session
                or not ui.authed(request.cookie)
                or request.client_ip not in {'127.0.0.1', '::1'}
                or not ui.host_ok(request.host_header)):
            raise Denied('a live local person session is required')
        headers = {key.lower(): value for key, value in request.handler.headers.items()}
        if boardroute._checked(request.method, headers,
                               request.handler.server.server_address[1], writes=True):
            raise Denied('the person request must come from this page')
        project = self._host.projects.get(self._project_id)
        if (not self._host.projects.hosts(project.board_host)
                or not request.path.startswith(f'/ui/projects/{self._project_id}/')):
            raise Denied('the selected project requires its local authority')
        self.registry._storage()

    def _person(self, agents):
        people = [(name, entry) for name, entry in agents.items() if entry.get('role') == HUMAN]
        if not people:
            raise SetupRequired('Join this workspace as person to use conversations and tasks.')
        if len(people) != 1:
            raise Denied('the workspace person binding is ambiguous')
        name, entry = people[0]
        if not self.registry._live(agents, entry):
            raise Denied('the workspace person is revoked or expired')
        binding = entry.get('person_project')
        if binding is None:
            raise SetupRequired('Join this workspace as person to use conversations and tasks.')
        if binding != self._project_id:
            raise Denied('the person belongs to another project')
        return Identity(name, HUMAN, str(entry.get('parent', '')),
                        tuple(cap for cap in ('read', 'send') if cap in entry.get('can', ('read', 'send'))))

    def connect(self):
        self._validate()
        if self._request.method != 'POST':
            raise Denied('joining a workspace requires an explicit person post')
        with held(self.registry.path.with_name('agents.lock')):
            self._validate()
            agents = self.registry._load()
            people = [(name, entry) for name, entry in agents.items() if entry.get('role') == HUMAN]
            if not people:
                name = 'local-person'
                if name in agents:
                    raise Denied('the person name is already registered')
                agents[name] = {'role': HUMAN, 'parent': '', 'can': ['read', 'send'],
                                'created': self.clock(), 'expires': 0, 'revoked': False,
                                'auth_method': 'browser-session', 'person_project': self._project_id}
            elif len(people) == 1:
                name, entry = people[0]
                if (not self.registry._live(agents, entry)
                        or entry.get('person_project') not in (None, self._project_id)):
                    raise Denied('the existing person cannot join this project')
                agents[name] = {**entry, 'person_project': self._project_id}
            else:
                raise Denied('the workspace person binding is ambiguous')
            self.registry._save(agents)
            return {'me': self._person(agents).id, 'project_id': self._project_id}

    def auth(self, token):
        if token is not self._actor:
            return super().auth(token)
        self._validate()
        return self._person(self.registry._load())

    def mint(self, *args, **kwargs):
        raise Denied('browser person sessions cannot mint credentials')

    def delegate(self, *args, **kwargs):
        raise Denied('browser person sessions cannot delegate credentials')
