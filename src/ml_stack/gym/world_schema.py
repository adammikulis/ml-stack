"""Native per-example world construction controls."""

TRAFFIC = {'modes': ['procedural', 'manual'], 'fields': {
    'arm_length': {'type': 'number', 'min': 60, 'max': 500, 'unit': 'metres'},
    'lanes': {'type': 'integer', 'min': 1, 'max': 2, 'default': 1},
    'vehicle_period': {'type': 'number', 'min': .5, 'max': 60, 'default': 3, 'unit': 'seconds'},
    'demand_seconds': {'type': 'integer', 'min': 60, 'max': 86400, 'default': 3600},
    'net_file': {'type': 'file', 'mode': 'manual', 'format': 'SUMO net.xml'},
    'route_file': {'type': 'file', 'mode': 'manual', 'format': 'SUMO routes XML'}},
    'constraint': 'One signal-controlled intersection; native SUMO road and demand generation'}

WAREHOUSE = {'modes': ['procedural', 'manual'], 'fields': {
    'shelf_columns': {'type': 'integer', 'choices': [3, 5, 7]},
    'shelf_rows': {'type': 'integer', 'min': 1, 'max': 4},
    'column_height': {'type': 'integer', 'min': 2, 'max': 8},
    'n_agents': {'type': 'integer', 'min': 1, 'max': 4, 'default': 2},
    'layout': {'type': 'textarea', 'mode': 'manual', 'format': 'RWARE ASCII: x shelves, . aisles, g goals'}},
    'constraint': 'Native RWARE lattice generator or rectangular ASCII layout'}
