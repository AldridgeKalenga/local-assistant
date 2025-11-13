# nav.py
# Navigation helpers for opening Apple Maps / Google Maps.

import sys
import urllib.parse
import webbrowser

def _is_latlon(text):
    try:
        lat, lon = [float(x.strip()) for x in text.split(",")]
        return True, lat, lon
    except Exception:
        return False, None, None

def open_maps_destination(value: str):
    """
    value can be "123 Main St, City" or "lat,lon".
    Opens default maps (Apple Maps on macOS, otherwise Google Maps).
    Returns the URL we attempted to open.
    """
    is_ll, lat, lon = _is_latlon(value)
    if sys.platform == "darwin":
        url = (
            f"http://maps.apple.com/?daddr={lat},{lon}"
            if is_ll else
            f"http://maps.apple.com/?daddr={urllib.parse.quote(value)}"
        )
    else:
        url = (
            f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
            if is_ll else
            f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(value)}"
        )
    webbrowser.open(url)
    return url
