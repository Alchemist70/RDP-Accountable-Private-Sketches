<#
PowerShell helper to publish a single image to GitHub Pages.
Requirements:
 - Git (git)
 - GitHub CLI (gh) authenticated: run `gh auth login` first
Usage example (from repo root):
 pwsh .\scripts\publish_image_github_pages.ps1 -GitHubUsername youruser -RepoName private-sketch-figs -ImagePath submission_package/privacy_utility_tradeoff_eps.png
This will create a new public repo `youruser/private-sketch-figs`, push a tiny repo with `docs/` containing the image, enable Pages with `main` branch + `/docs` source, and output the expected public URL.
#>
param(
    [Parameter(Mandatory=$true)][string]$GitHubUsername,
    [Parameter(Mandatory=$true)][string]$RepoName,
    [Parameter(Mandatory=$false)][string]$ImagePath = "submission_package/privacy_utility_tradeoff_eps.png",
    [switch]$Public
)

function Fail([string]$msg){ Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Fail "GitHub CLI 'gh' not found. Install from https://cli.github.com/ and run 'gh auth login'." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git not found. Install git and try again." }
if (-not (Test-Path $ImagePath)) { Fail "ImagePath not found: $ImagePath" }

$tmp = Join-Path $env:TEMP ("gh-pages-upload-" + [System.Guid]::NewGuid().ToString())
New-Item -Path $tmp -ItemType Directory -Force | Out-Null
$docs = Join-Path $tmp "docs"
New-Item -Path $docs -ItemType Directory -Force | Out-Null

$destImageName = Split-Path $ImagePath -Leaf
Copy-Item -Path $ImagePath -Destination (Join-Path $docs $destImageName) -Force

Push-Location $tmp
try {
    git init -b main | Out-Null
    git add .
    git commit -m "Add image for GitHub Pages" | Out-Null

    Write-Host "Creating remote repo $GitHubUsername/$RepoName and pushing..."
    gh repo create "$GitHubUsername/$RepoName" --public --source . --remote origin --push --confirm | Out-Null

    Write-Host "Enabling GitHub Pages (branch: main, path: /docs)..."
    gh api -X POST "/repos/$GitHubUsername/$RepoName/pages" -f "source[branch]=main" -f "source[path]=/docs" | Out-Null

    Start-Sleep -Seconds 3

    $url = "https://$GitHubUsername.github.io/$RepoName/$destImageName"
    Write-Host "Published. Expected public URL:" -ForegroundColor Green
    Write-Host $url -ForegroundColor Cyan
    Write-Host "Note: GitHub Pages may take a minute to propagate. If the URL 404s, wait ~1-2 minutes and retry." -ForegroundColor Yellow
}
catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Fail "Publishing failed. Inspect output above." 
}
finally {
    Pop-Location
}

# End
