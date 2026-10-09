"""RWARE native lattice parameters and manual ASCII warehouse definitions."""

from importlib.metadata import version

from poolhouse.gym.world_files import directory, record


def integer(value, low, high, name):
    parsed = int(value)
    if parsed != float(value) or not low <= parsed <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return parsed


def build(spec=None, seed=0):
    import gymnasium as gym
    import numpy as np
    import rware  # noqa: F401

    spec = dict(spec or {})
    mode, seed = spec.pop('mode', 'procedural'), int(spec.pop('seed', seed))
    if not 0 <= seed < 2**31:
        raise ValueError('World seed must be between zero and 2147483647')
    agents = integer(spec.pop('n_agents', 2), 1, 4, 'n_agents')
    if mode == 'procedural':
        rng = np.random.default_rng(seed)
        columns = integer(spec.pop('shelf_columns', int(rng.choice([3, 5, 7]))), 3, 7, 'shelf_columns')
        if columns % 2 != 1:
            raise ValueError('RWARE shelf_columns must be odd')
        native = {'shelf_columns': columns,
                  'shelf_rows': integer(spec.pop('shelf_rows', int(rng.integers(1, 4))), 1, 4, 'shelf_rows'),
                  'column_height': integer(spec.pop('column_height', int(rng.integers(2, 6))), 2, 8, 'column_height'),
                  'n_agents': agents}
    elif mode == 'manual':
        layout = spec.pop('layout', '').strip().replace(' ', '').lower()
        lines = layout.splitlines()
        if not lines or len(lines) > 100 or not 1 <= len(lines[0]) <= 100:
            raise ValueError('RWARE layout must be between 1 and 100 cells per side')
        if any(len(line) != len(lines[0]) or set(line) - set('x.g') for line in lines):
            raise ValueError('RWARE layout must be rectangular and use x, . and g only')
        if 'g' not in layout or layout.count('x') < agents or layout.count('.') < agents:
            raise ValueError('RWARE layout needs goals, shelves and enough aisle cells for its robots')
        native = {'layout': layout, 'n_agents': agents}
    else:
        raise ValueError('World mode must be procedural or manual')
    if spec:
        raise ValueError(f'Unknown warehouse world settings: {sorted(spec)}')
    definition = {'mode': mode, 'seed': seed, 'rware_version': version('rware'), **native}
    path = directory('rware', definition)
    file = path / 'layout.txt'
    if mode == 'manual':
        file.write_text(native['layout'] + '\n')
    else:
        env = gym.make('rware-tiny-2ag-v2', **native)
        try:
            warehouse = env.unwrapped
            rows = [['.' if cell else 'x' for cell in row] for row in warehouse.highways]
            for x, y in warehouse.goals:
                rows[y][x] = 'g'
            file.write_text('\n'.join(''.join(row) for row in rows) + '\n')
        finally:
            env.close()
    return native, record(path, 'RWARE native lattice' if mode == 'procedural' else 'RWARE ASCII',
                          definition, {'layout': file})
