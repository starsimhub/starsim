#!/usr/bin/env python3
"""
Copy Claude Code transcripts for this repo into transcripts/, and extract the
prompts that are not yet recorded in prompts.md.

Usage:
    copy_prompts.py status              # List sessions and how many prompts are new
    copy_prompts.py copy [--dry-run]    # Copy new or changed transcripts
    copy_prompts.py extract [id ...]    # Print new prompts and the replies to them
"""

import re
import sys
import json
import shutil
import collections
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
source = Path.home() / '.claude' / 'projects' / re.sub(r'[^a-zA-Z0-9]', '-', str(repo))
dest = repo / 'transcripts'
prompts_md = repo / 'prompts.md'

# User-role messages that were not typed by the user
skip_prefixes = (
    '<local-command',
    '<command-',
    '<system-reminder>',
    '<task-notification>',
    '[Request interrupted',
    'This session is being continued from a previous conversation',
)


def get_text(entry):
    """ Get the text of a transcript entry, ignoring tool calls and results """
    content = entry.get('message', {}).get('content', '')
    if isinstance(content, str):
        return content
    return '\n'.join(part.get('text', '') for part in content if part.get('type') == 'text')


def is_prompt(entry, text):
    """ Check whether a transcript entry is a prompt typed by the user """
    if entry.get('type') != 'user' or entry.get('isMeta') or entry.get('isSidechain') or entry.get('isCompactSummary'):
        return False
    text = text.strip()
    if not text or text.startswith(skip_prefixes) or re.fullmatch(r'/\S+', text):
        return False
    return True


def load_session(path):
    """ Return a list of dicts (time, prompt, reply) for each prompt in a transcript """
    turns = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = get_text(entry)
        if is_prompt(entry, text):
            turns.append(dict(time=entry.get('timestamp', ''), prompt=text, reply=''))
        elif entry.get('type') == 'assistant' and not entry.get('isSidechain') and text.strip() and turns:
            turns[-1]['reply'] = text # Keep the last text reply before the next prompt
    return turns


def load_sessions():
    """ Load all sessions that contain at least one prompt, oldest first """
    sessions = {}
    for path in source.glob('*.jsonl'):
        turns = load_session(path)
        if turns:
            sessions[path] = turns
    return dict(sorted(sessions.items(), key=lambda item: item[1][0]['time']))


def mark_new(sessions):
    """ Flag each prompt as new if its exact text is not in a code block in prompts.md """
    recorded = collections.Counter()
    if prompts_md.exists():
        blocks = re.findall(r'^```text\n(.*?)\n```$', prompts_md.read_text(), flags=re.DOTALL | re.MULTILINE)
        recorded.update(block.strip() for block in blocks)
    for turns in sessions.values():
        for turn in turns:
            key = turn['prompt'].strip()
            turn['new'] = recorded[key] == 0
            if not turn['new']:
                recorded[key] -= 1
    return sessions


def needs_copy(src, dst):
    """ Check whether a transcript is missing from or newer than the copy """
    if not dst.exists():
        return True
    s, d = src.stat(), dst.stat()
    return s.st_size != d.st_size or s.st_mtime > d.st_mtime


def status():
    sessions = mark_new(load_sessions())
    print(f'Source: {source}\n')
    print(f'{"session":38s} {"started":18s} {"prompts":>7s} {"new":>4s}  copied')
    for path, turns in sessions.items():
        n_new = sum(turn['new'] for turn in turns)
        copied = 'stale' if needs_copy(path, dest / path.name) else 'yes'
        if not (dest / path.name).exists():
            copied = 'no'
        print(f'{path.stem:38s} {turns[0]["time"][:16]:18s} {len(turns):7d} {n_new:4d}  {copied}')


def copy(dry_run=False):
    verb = 'Would copy' if dry_run else 'Copied'
    n = 0
    for path in load_sessions():
        if not needs_copy(path, dest / path.name):
            continue
        n += 1
        if not dry_run:
            dest.mkdir(exist_ok=True)
            shutil.copy2(path, dest / path.name)
            extras = path.with_suffix('') # Subagent logs and tool results, if any
            if extras.is_dir():
                shutil.copytree(extras, dest / extras.name, dirs_exist_ok=True)
        print(f'{verb} {path.name}')
    print(f'{verb} {n} transcript(s) to {dest}')


def extract(ids):
    sessions = mark_new(load_sessions())
    for path, turns in sessions.items():
        if ids and not any(path.stem.startswith(i) for i in ids):
            continue
        turns = [turn for turn in turns if turn['new'] or ids]
        if not turns:
            continue
        print(f'{"="*80}\nSESSION {path.name}\n{"="*80}')
        for turn in turns:
            print(f'\n----- PROMPT [{turn["time"]}] -----\n{turn["prompt"]}')
            print(f'\n----- REPLY -----\n{turn["reply"]}')


if __name__ == '__main__':
    args = sys.argv[1:]
    command = args[0] if args else 'status'
    if command == 'status':
        status()
    elif command == 'copy':
        copy(dry_run='--dry-run' in args)
    elif command == 'extract':
        extract(args[1:])
    else:
        sys.exit(__doc__)
