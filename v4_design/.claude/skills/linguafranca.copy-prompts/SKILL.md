---
name: linguafranca.copy-prompts
description: Use when the user asks to update prompts.md, record or extract the prompts from recent sessions, or copy/archive new Claude Code transcripts for the lingua franca design repo.
---

# Copy prompts and transcripts

The design docs (`v4_design/` in the Starsim repo, formerly the `lingua_franca_design` repo) keep a record of the prompts used to build them and Starsim v4 in `admin/prompts.md`, and an archive of the full Claude Code session transcripts in `admin/transcripts/` (gitignored). This skill brings both up to date. The helper script `copy_prompts.py`, in the same folder as this file, does the mechanical work; it uses only the Python standard library and finds everything by itself. It reads the sessions of both folders the work is done in, the Starsim checkout and `lingua_franca_design` next to it (`~/.claude/projects/<folder path with non-alphanumerics replaced by ->/`).

## Steps

1. From `v4_design/`, run `python3 .claude/skills/linguafranca.copy-prompts/copy_prompts.py status` to list every session with its number of prompts, how many are not yet in `prompts.md`, and whether the transcript has been copied.
2. Run `copy_prompts.py copy` to copy new or changed transcripts into `admin/transcripts/`, including each session's folder of subagent logs if it has one. Sessions with no typed prompts (e.g. only `/clear`) are skipped. The current session is copied too; it will show as stale and be refreshed on the next run. Use `copy --dry-run` to preview.
3. Run `copy_prompts.py extract` to print each prompt that is not yet in `prompts.md`, followed by the final reply to it (ignoring a closing note on staged lessons). If that reply is only a short aside, look further back in the transcript for the real report. To print every prompt in particular sessions instead, pass session IDs or ID prefixes (e.g. `extract 3315bd22`). The output can be long, so redirect it to a scratch file and read that.
4. Add the new prompts to `prompts.md`, following the format already there:
   - One `## Session N: <topic>` section per session, with a `Transcript: \`<id>.jsonl\`` line. If a session already has a section, add the new prompts to it.
   - One `### Prompt N (HH:MM)` heading per prompt, numbered consecutively across the whole document, with the time in UTC. Add the date wherever it differs from the session before.
   - The prompt in a ` ```text ` fenced block, copied exactly as typed, typos included. Do not tidy it. The script recognizes a prompt as recorded by matching the text of these blocks exactly, so a changed prompt will be reported as new again.
   - A `**Response.**` paragraph, plus bullets where useful, summarizing what the reply said and which files were created or changed. Write the summary from the extracted reply and from `git log --stat`; do not copy the reply wholesale.
   - A new row in the summary table at the top, with the commit that holds the result if there is one, marked `Starsim` or `LFD` for the repo it is in.
5. Rerun `status` and check that the `new` column is zero for every session that was recorded.

## Judgement calls

- Ask the user before recording prompts that are not about the repo's content (e.g. housekeeping, questions about where they got up to, requests to run this skill, or managing Claude's own notes), unless they have already said whether they want those included. Prompts from the same folders about other projects belong to those projects and are not recorded.
- Prompts that only make sense together (a request and its short follow-up) still get separate headings, as in the existing document.
- Do not commit; leave `prompts.md` modified in the working tree.
