#!/usr/bin/env python
"""
Manually add aliases to Starsim functions, so instead of starsim.analyzers.Analyzer, you can link via starsim.Analyzer (and therefore ss.Analyzer)
"""
print('Customizing aliases ...')
import sciris as sc

# Get everything in the Starsim namespace -- this is the only part that needs to be customized
import starsim
mod_name = 'starsim'
mod_items = dir(starsim)

# Load the current objects inventory
filename = 'objects.json'
json = sc.loadjson(filename)
items = json['items']
names = [item['name'] for item in items]
print(f'  Loaded {len(json["items"])} items')

# Collect duplicates
dups = []
for item in items:
    parts = item['name'].split('.')
    if len(parts) < 3 or parts[0] != mod_name:
        continue
    objname = parts[2] # e.g. 'Analyzer' from starsim.analyzers.Analyzer
    if objname in mod_items:
        remainder = '.'.join(parts[2:])
        alias = f'{mod_name}.{remainder}'
        if alias not in names:
            dup = sc.dcp(item)
            dup['name'] = alias
            dups.append(dup)

# Add them back into the JSON
items.extend(dups)

# Save the modified version
sc.savejson(filename, json)
print(f'  Saved {len(json["items"])} items')