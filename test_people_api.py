#!/usr/bin/env python3
"""
Quick test script for Google People API integration.
Tests if we can fetch contact addresses.
"""

from people_tools import (
    PEOPLE_AVAILABLE,
    people_get_contact_addresses,
    people_sync_to_places,
)
from profiles import load_profiles

print("=== Google People API Test ===\n")

# Check if API is available
if not PEOPLE_AVAILABLE:
    print("Google People API not available.")
    print("   Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    exit(1)

print("Google People API libraries available\n")

# Test fetching contacts
print("1. Testing contact fetch...")
try:
    contacts = people_get_contact_addresses(identity="Aldridge")
    
    if contacts:
        print(f"Found {len(contacts)} contacts with addresses:")
        for name, address_list in contacts.items():
            print(f"   - {name}: {len(address_list)} address(es)")
            for addr, addr_type in address_list:
                print(f"     → [{addr_type}] {addr}")
    else:
        print("No contacts with addresses found.")
        print("   Make sure you have addresses saved in your Google Contacts.")
        print("   Go to: https://contacts.google.com")
    
    print("\n2. Testing sync to profiles...")
    profiles = load_profiles()
    synced, msg = people_sync_to_places(identity="Aldridge", profiles=profiles)
    print(f"{msg}")
    
    if synced > 0:
        print("\n3. Checking synced places...")
        from profiles import load_profiles
        profiles = load_profiles()
        places = profiles.get("Aldridge", {}).get("places", {})
        if places:
            print("Synced places:")
            for k, v in places.items():
                print(f"   - {k}: {v}")
        else:
            print("No places in profiles after sync")
    else:
        print("\nNo addresses were synced.")
        print("   This could mean:")
        print("   - No contacts have addresses in Google Contacts")
        print("   - Addresses already exist in profiles.json")
        print("   - OAuth token needs to be refreshed")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure People API is enabled in Google Cloud Console")
    print("2. Make sure OAuth consent screen includes contacts.readonly scope")
    print("3. You may need to re-authenticate (delete token_aldridge.json and try again)")
    import traceback
    traceback.print_exc()

