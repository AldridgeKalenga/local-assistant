# people_tools.py
# Google People API integration for pulling contact addresses.

import os
from config import (
    GCAL_SCOPES,
    GCAL_CREDENTIALS_PATH,
    CONTACT_NAME_MAP,
)

# Try to import Google People API libs
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    PEOPLE_AVAILABLE = True
except Exception:
    PEOPLE_AVAILABLE = False

def people_build_service(identity="Aldridge"):
    """
    Build a Google People API service client for a specific persona.
    Uses same token system as calendar (token_<identity>.json).
    """
    if not PEOPLE_AVAILABLE:
        raise RuntimeError("Google People API libs not installed.")

    token_path = f"token_{identity.lower()}.json"
    creds_path = f"credentials_{identity.lower()}.json"
    if not os.path.exists(creds_path):
        creds_path = GCAL_CREDENTIALS_PATH

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GCAL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, GCAL_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("people", "v1", credentials=creds)

def _get_contact_name_for_identity(identity):
    """
    Get the Google Contact name to search for based on identity.
    Uses CONTACT_NAME_MAP if available, otherwise uses identity name.
    Returns the contact name to search for.
    """
    # Check if there's a mapping in config
    if identity in CONTACT_NAME_MAP:
        return CONTACT_NAME_MAP[identity]
    # Default: use identity name itself
    return identity

def people_get_contact_addresses(identity="Aldridge"):
    """
    Fetch all contacts with addresses from Google People API.
    Returns: {contact_name: [(address, type), ...]}
    Only pulls names and addresses (no email, phone, etc.)
    Returns address type if available (e.g., "home", "work", "other")
    """
    if not PEOPLE_AVAILABLE:
        return {}

    try:
        service = people_build_service(identity=identity)
        contacts = {}
        page_token = None
        
        # Handle pagination to get ALL contacts
        while True:
            # Build request parameters
            request_params = {
                'resourceName': 'people/me',
                'personFields': 'names,addresses',  # Only what we need
                'pageSize': 100  # Max per page
            }
            
            # Add page token if we're continuing from a previous page
            if page_token:
                request_params['pageToken'] = page_token
            
            results = service.people().connections().list(**request_params).execute()
            
            # Process contacts from this page
            for person in results.get('connections', []):
                # Get display name
                names = person.get('names', [])
                if not names:
                    continue
                display_name = names[0].get('displayName', '').strip()
                if not display_name:
                    continue

                # Get all addresses for this contact
                addresses = person.get('addresses', [])
                if not addresses:
                    continue

                # Extract formatted addresses with their types
                contact_addresses = []
                for addr in addresses:
                    formatted = addr.get('formattedValue', '').strip()
                    if formatted:
                        # Get address type (home, work, other, etc.)
                        addr_type = addr.get('type', 'other').lower()
                        # Normalize common types
                        if addr_type not in ['home', 'work', 'other']:
                            addr_type = 'other'
                        contact_addresses.append((formatted, addr_type))

                if contact_addresses:
                    contacts[display_name] = contact_addresses
            
            # Check if there are more pages
            page_token = results.get('nextPageToken')
            if not page_token:
                break  # No more pages

        return contacts

    except Exception as e:
        print(f"(People API error: {e})")
        return {}

def people_sync_to_places(identity="Aldridge", profiles=None):
    """
    Sync Google Contacts addresses to profiles.json places.
    If a contact name matches the identity (e.g., "Aldridge"), uses simple keys like "home", "work".
    For other contacts, uses keys like "john_smith", "john_smith_home", "john_smith_work".
    Returns: (synced_count, message)
    """
    if profiles is None:
        from profiles import load_profiles
        profiles = load_profiles()

    contacts = people_get_contact_addresses(identity=identity)
    if not contacts:
        return 0, "No contacts with addresses found. Make sure you have addresses saved in your Google Contacts."

    from profiles import ensure_identity_struct, save_profiles
    ensure_identity_struct(profiles, identity)

    synced = 0
    # Get the contact name to search for (may differ from identity)
    contact_name_to_find = _get_contact_name_for_identity(identity)
    contact_name_lower = contact_name_to_find.lower().strip()
    identity_lower = identity.lower().strip()
    
    for name, address_list in contacts.items():
        # Check if this contact matches the current identity
        # First try exact match with the contact name we're looking for
        name_lower = name.lower().strip()
        name_parts = name_lower.split()
        contact_name_parts = contact_name_lower.split()
        
        # Check for exact match with contact name
        is_my_contact = (name_lower == contact_name_lower)
        
        # Also check if contact name matches the first name(s) of the contact
        # e.g., "Aldridge Kalenga" matches "Aldridge Kalenga" or "Aldridge Kalenga Smith"
        if not is_my_contact and contact_name_parts:
            # Check if the first part(s) of the contact name match
            if len(name_parts) >= len(contact_name_parts):
                contact_start = ' '.join(name_parts[:len(contact_name_parts)])
                if contact_start == contact_name_lower:
                    is_my_contact = True
        
        # Fallback: also check if identity name matches (for backwards compatibility)
        # e.g., if identity is "Aldridge" and contact is "Aldridge Kalenga"
        if not is_my_contact:
            identity_parts = identity_lower.split()
            if identity_parts and len(name_parts) >= len(identity_parts):
                contact_start = ' '.join(name_parts[:len(identity_parts)])
                if contact_start == identity_lower:
                    is_my_contact = True
        
        if is_my_contact:
            # For the user's own contact, use simple keys: "home", "work", "school", etc.
            for addr, addr_type in address_list:
                # Use the address type from Google Contacts, or fallback to simple numbering
                if addr_type in ['home', 'work', 'other']:
                    if addr_type == 'other':
                        # For "other" type, try to use a more descriptive key
                        # If only one address, use "home", otherwise number them
                        if len(address_list) == 1:
                            key = "home"
                        else:
                            # Find index to create unique key
                            idx = [a[0] for a in address_list].index(addr)
                            key = f"address_{idx + 1}" if idx > 0 else "home"
                    else:
                        key = addr_type  # "home" or "work"
                else:
                    # Fallback: use index-based naming
                    idx = [a[0] for a in address_list].index(addr)
                    key = "home" if idx == 0 else f"address_{idx + 1}"
                
                if key not in profiles[identity]["places"]:
                    profiles[identity]["places"][key] = addr
                    synced += 1
        else:
            # For other contacts, use contact name as key base
            key_base = name.lower().replace(' ', '_').replace("'", "").replace(".", "")

            # If only one address, use just the name
            if len(address_list) == 1:
                key = key_base
                if key not in profiles[identity]["places"]:
                    profiles[identity]["places"][key] = address_list[0][0]  # Get address string
                    synced += 1
            else:
                # Multiple addresses: create keys like "john_smith_home", "john_smith_work"
                for addr, addr_type in address_list:
                    # Use address type if it's meaningful, otherwise use index
                    if addr_type in ['home', 'work']:
                        key = f"{key_base}_{addr_type}"
                    else:
                        idx = [a[0] for a in address_list].index(addr)
                        key = f"{key_base}_home" if idx == 0 else f"{key_base}_address_{idx + 1}"
                    
                    if key not in profiles[identity]["places"]:
                        profiles[identity]["places"][key] = addr
                        synced += 1

    if synced == 0:
        return 0, "No new addresses to sync (all addresses already exist in profiles)."

    if synced > 0:
        save_profiles(profiles)

    return synced, f"Synced {synced} addresses from Google Contacts."

