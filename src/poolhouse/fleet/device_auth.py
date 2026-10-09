"""Request credentials for active paired devices."""

import base64
import hmac

from poolhouse import macauth


def secret(device):
    if device.status != 'active' or not device.secret:
        return ''
    try:
        return macauth.derive(base64.urlsafe_b64decode(device.secret))
    except (ValueError, TypeError):
        return ''


def identify(devices, credential):
    matches = [device for device in devices if secret(device)
               and hmac.compare_digest(secret(device), credential)]
    return matches[0] if len(matches) == 1 else None
