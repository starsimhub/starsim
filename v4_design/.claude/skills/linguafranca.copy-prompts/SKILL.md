---
name: linguafranca.copy-prompts
description: Use when the user asks to update prompts.md, record or extract the prompts from recent sessions, or copy/archive new Claude Code transcripts for the lingua franca design repo.
---

# Copy prompts and transcripts

This repo keeps a record of the prompts used to build it in `prompts.md`, and an archive of the full Claude Code session transcripts in `transcripts/` (gitignored). This skill brings both up to date. The helper script `copy_prompts.py`, in the same folder as this file, does the mechanical work; it uses only the Python standard library and locates the repo and its transcript folder (`~/.claude/projects/<repo path with non-alphanumerics replaced by ->/`) by itself.

## Steps

1. Run `python3 .claude/skills/linguafranca.copy-prompts/copy_prompts.py status` to list every session with its number of prompts, how many are not yet in `prompts.md`, and whether the transcript has been copied.
2. Run `copy_prompts.py copy` to copy new or changed transcripts into `transcripts/`, including each session's folder of subagent logs if it has one. Sessions with no typed prompts (e.g. only `/clear`) are skipped. The current session is copied too; it will show as stale and be refreshed on the next run. Use `copy --dry-run` to preview.
3. Run `copy_prompts.py extract` to print each prompt that is not yet in `prompts.md`, followed by the final reply to it. To print every prompt in particular sessions instead, pass session IDs or ID prefixes (e.g. `extract 3315bd22`). The output can be long, so redirect it to a scratch file and read that.
4. Add the new prompts to `prompts.md`, following the format already there:
   - One `## Session N: <topic>` section per session, with a `Transcript: \`<id>.jsonl\`` line. If a session already has a section, add the new prompts to it.
   - One `### Prompt N (HH:MM)` heading per prompt, numbered consecutively across the whole document, with the time in UTC. Add the date wherever it differs from the session before.
   - The prompt in a ` ```text ` fenced block, copied exactly as typed, typos included. Do not tidy it. The script recognizes a prompt as recorded by matching the text of these blocks exactly, so a changed prompt will be reported as new again.
   - A `**Response.**` paragraph, plus bullets where useful, summarizing what the reply said and which files were created or changed. Write the summary from the extracted reply and from `git log --stat`; do not copy the reply wholesale.
   - A new row in the summary table at the top, with the commit that holds the result if there is one.
5. Rerun `status` and check that the `new` column is zero for every session that was recorded.

## Judgement calls

- Ask the user before recording prompts that are not about the repo's content (e.g. housekeeping, questions about where they got up to, or requests to run this skill), unless they have already said whether they want those included.
- Prompts that only make sense together (a request and its short follow-up) still get separate headings, as in the existing document.
- Do not commit; leave `prompts.md` modified in the working tree.
