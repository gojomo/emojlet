# except this comment line, this file comes verbatim from ChatGPT-4 (2023-06-07)
import urllib.request
import re
import os
import datetime

def download_and_parse_unicode_age_data(path='unicode/DerivedAge.txt'):
    """Download and parse DerivedAge.txt, save it locally, and return a dictionary mapping characters to Unicode versions."""
    url = "https://www.unicode.org/Public/UCD/latest/ucd/DerivedAge.txt"

    # Check if local file exists and is less than a month old
    if os.path.exists(path):
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        if (datetime.datetime.now() - file_time) < datetime.timedelta(days=30):
            with open(path, 'r', encoding='utf-8') as file:
                data = file.read()
        else:
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            with open(path, 'w', encoding='utf-8') as file:
                file.write(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        with open(path, 'w', encoding='utf-8') as file:
            file.write(data)

    age_data = {}
    for line in data.split('\n'):
        if not line or line.startswith('#'):
            continue
        m = re.match(r"([0-9A-F]+)(?:..([0-9A-F]+))?\s+;\s+(\d+\.\d+)", line)
        if m:
            start, end, version = m.groups()
            for codepoint in range(int(start, 16), int(end or start, 16) + 1):
                age_data[chr(codepoint)] = float(version)
    return age_data

# Create the lookup dictionary
unicode_age_data = download_and_parse_unicode_age_data()

def get_unicode_version(character):
    """Given a character, return the Unicode version it first appeared in as a float, or float('inf') if not found."""
    return unicode_age_data.get(character, float('inf'))
