param(
    [string]$GitHubUser, FelipeT
    [string]$RepoName = "NOSSO-MAR",
    [string]$Token
)

if (-not $GitHubUser) {
    Write-Error "GitHub username is required. Example: .\push_to_github.ps1 -GitHubUser <user> -Token <token> [-RepoName <name>]"
    exit 2
}
if (-not $Token) {
    Write-Error "GitHub token is required. Create a PAT with 'repo' scope and pass it as -Token.";
    exit 2
}

$remoteUrl = "https://$GitHubUser:$Token@github.com/$GitHubUser/$RepoName.git"

Write-Host "Setting remote origin to" $remoteUrl

git remote remove origin -f 2>$null | Out-Null
git remote add origin $remoteUrl

# Push main branch
Write-Host "Pushing 'main' branch to origin (force if necessary)..."
$pushResult = git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "Initial push failed. Attempting to create 'main' branch and push."
    git branch -M main
    git push -u origin main --force
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Push succeeded. Remote: https://github.com/$GitHubUser/$RepoName"
} else {
    Write-Error "Push failed. Inspect output above for details."
}
