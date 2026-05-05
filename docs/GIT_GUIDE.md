# Git & GitHub — Practical Guide

A quick reference for working with Git and GitHub using VS Code. Keep this file in your repository root and consult it whenever needed.

---

## One-Time Setup

Before using Git for the first time, configure your identity[cite:91][cite:94]:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Confirm it worked:

```bash
git config --list
```

---

## Opening the Repository in VS Code

If the folder is already on your machine, open the terminal inside VS Code with `Ctrl + '` and run[cite:2][cite:13]:

```bash
cd path/to/project
code .
```

Or open VS Code manually via **File > Open Folder**.

---

## Command Reference

| Command | What it does |
|--------|--------------|
| `git status` | Shows modified, staged, and untracked files |
| `git pull` | Downloads remote changes and merges them into the local branch |
| `git fetch` | Downloads remote info without touching local files |
| `git add .` | Stages all modified and new files for the next commit |
| `git add <file>` | Stages a specific file only |
| `git commit -m "..."` | Saves staged changes as a snapshot in local history |
| `git push` | Sends local commits to GitHub |
| `git push -u origin <branch>` | First push of a new branch |
| `git stash` | Temporarily saves local changes without committing |
| `git stash pop` | Restores previously stashed changes |
| `git checkout -b <branch>` | Creates and switches to a new branch |
| `git checkout <branch>` | Switches to an existing branch |
| `git branch` | Lists all local branches |
| `git diff` | Shows line-by-line differences in modified files |
| `git log --oneline -5` | Shows the last 5 commits in compact format |
| `git reset --soft HEAD~1` | Undoes the last commit, keeps changes staged |
| `git reset --hard HEAD~1` | Undoes the last commit and discards all changes (irreversible) |

---

## Daily Workflow

Follow this sequence every time you sit down to work[cite:27][cite:42]:

```bash
git status        # check current state
git pull          # sync with remote (only if status is clean)
# ... edit files ...
git status        # confirm what changed
git diff          # review changes line by line (optional)
git add .         # stage everything
git commit -m "Clear description of what was done"
git push          # send to GitHub
```

---

## Scenario 1 — No Local Changes, Remote Has Updates

You have not edited anything locally. Someone (or you, via the GitHub web editor) changed files on GitHub[cite:27][cite:37]:

```bash
git status        # should show "nothing to commit, working tree clean"
git pull          # bring down remote changes
git status        # confirm you're up to date
```

---

## Scenario 2 — You Edited Locally After the Last Push, Remote Also Changed

You have unpushed local commits and GitHub also has new changes since your last push[cite:42][cite:51]:

```bash
git status        # see what you changed
git add .         # stage your changes
git commit -m "Your work description"
git pull          # merge remote changes into your branch
# resolve conflicts if any appear
git push          # send everything to GitHub
```

---

## Scenario 3 — You Edited Locally Without Pushing, Remote Also Changed (No Commit Yet)

You have local edits that are not yet committed and GitHub has new changes[cite:35][cite:37]:

```bash
git status        # confirm you have unstaged changes
git stash         # temporarily save your work
git pull          # update with remote changes
git stash pop     # restore your saved work
# resolve conflicts if any appear
git add .
git commit -m "Your work description"
git push
```

---

## Working with Branches

Working in separate branches avoids breaking the main codebase[cite:88][cite:93]:

```bash
# Create and switch to a new branch
git checkout -b feature/task-name

# ... do your work, add and commit normally ...

# Push the new branch to GitHub for the first time
git push -u origin feature/task-name

# Switch back to main when done
git checkout main
```

Use descriptive names like `feature/wave-farm-model`, `fix/data-parser-bug` or `docs/update-readme`.

---

## Good Commit Messages

A commit message should answer: *what changed and why?*[cite:24][cite:89]

```
# Good examples
Add wave energy data preprocessing module
Fix boundary condition in SWAN model output parser
Update .gitignore to exclude temporary NetCDF files
Remove hardcoded path in configuration loader

# Bad examples
update
fix
changes
test
```

Keep messages short (under 72 characters), use the imperative form ("Add", "Fix", "Remove", "Update").

---

## .gitignore

A `.gitignore` file tells Git which files and folders to ignore completely[cite:24][cite:86]. Create one in the project root:

```
# Python
__pycache__/
*.pyc
*.pyo
.env

# Output and logs
output/
*.log
*.tmp

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.DS_Store

# Sensitive data
*.key
secrets.yaml
credentials.json
```

After saving, run `git status` to confirm those files no longer appear as untracked.

---

## Undoing Changes

```bash
# Discard local edits in a specific file (irreversible)
git checkout -- filename.py

# Undo last commit but keep changes staged
git reset --soft HEAD~1

# Undo last commit and discard all changes (irreversible)
git reset --hard HEAD~1
```

---

## Quick Rules

- Always run `git status` before and after any Git operation[cite:42][cite:44]
- Pull before you push if others (or you, elsewhere) may have changed the remote[cite:27][cite:86]
- One commit per logical change — not one commit per session[cite:24][cite:89]
- Never commit `.env`, passwords, API keys, or large binary files[cite:24][cite:92]
- When in doubt, use a branch[cite:88][cite:93]

