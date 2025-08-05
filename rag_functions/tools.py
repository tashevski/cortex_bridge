# tools.py

import dateutil.parser
from datetime import datetime

def perform_calculations(parsed_text):
    """
    Dummy example: extract dates, compute durations, interest, etc.
    """
    # Example: look for 'effective_date' and 'expiry_date' in parsed_text
    # and compute duration days between them.
    # This should be adapted to your semantic extraction output.
    lines = parsed_text.splitlines()
    def find_date(label):
        for l in lines:
            if label in l.lower():
                try:
                    return dateutil.parser.parse(l.split(":")[-1].strip())
                except:
                    continue
        return None

    d1 = find_date("effective_date")
    d2 = find_date("expiry_date")
    result = {}
    if d1 and d2:
        days = (d2 - d1).days
        result['duration_days'] = days
    else:
        result['duration_days'] = None
    return result
