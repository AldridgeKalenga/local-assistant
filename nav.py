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
    # Normalize address: replace newlines with spaces and clean up whitespace
    normalized = ' '.join(value.split())
    
    is_ll, lat, lon = _is_latlon(normalized)
    if sys.platform == "darwin":
        url = (
            f"http://maps.apple.com/?daddr={lat},{lon}"
            if is_ll else
            f"http://maps.apple.com/?daddr={urllib.parse.quote(normalized)}"
        )
    else:
        url = (
            f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
            if is_ll else
            f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(normalized)}"
        )
    webbrowser.open(url)
    return url
