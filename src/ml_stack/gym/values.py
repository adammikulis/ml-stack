"""JSON conversion for native simulator values."""

def json_value(value):
    """Convert simulator values into JSON values."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

