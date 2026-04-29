Push your local NOSSO-MAR repository to GitHub

Option A — Run the provided PowerShell script (recommended on Windows):

1. Create a GitHub Personal Access Token (PAT) with `repo` scope.
2. From a PowerShell prompt in `D:\Projects\NOSSO-MAR` run:

```powershell
# replace <user> and <token> with your values
.\push_to_github.ps1 -GitHubUser <your-gh-username> -RepoName NOSSO-MAR -Token <your-token>
```

The script will set `origin` to `https://<user>:<token>@github.com/<user>/NOSSO-MAR.git` and push the `main` branch.

Option B — Manual steps (no script):

```powershell
# from D:\Projects\NOSSO-MAR
git remote remove origin -f  # optional, only if an origin exists
git remote add origin https://<USER>:<TOKEN>@github.com/<USER>/NOSSO-MAR.git
git branch -M main
git push -u origin main
```

Security notes:
- Embedding PAT in remote URL is convenient but leaves it in your shell history; prefer using a credential helper (Windows Credential Manager) or GitHub CLI `gh auth login`.
- If you prefer not to expose the token, run the manual commands and enter credentials interactively, or use `gh repo create <USER>/NOSSO-MAR --public` then `git push`.

If you want, I can attempt to push from this environment provided you paste the PAT here (not recommended). Otherwise run the script locally and tell me the outcome; I can help fix any errors.
