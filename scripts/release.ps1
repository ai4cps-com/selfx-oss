[CmdletBinding()]
param(
    [string]$Version,

    [ValidateSet("patch", "minor", "major")]
    [string]$Bump = "patch",

    [string]$Remote = "origin",

    [string]$Branch = "main",

    [switch]$NoPush,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    & git -c "safe.directory=$gitSafeDirectory" @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-GitOutput {
    $output = & git -c "safe.directory=$gitSafeDirectory" @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
    return $output
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitSafeDirectory = $repoRoot -replace "\\", "/"
Set-Location $repoRoot

$insideWorkTree = (Get-GitOutput rev-parse --is-inside-work-tree).Trim()
if ($insideWorkTree -ne "true") {
    throw "This script must be run inside a git repository."
}

$currentBranch = (Get-GitOutput branch --show-current).Trim()
if ($currentBranch -ne $Branch) {
    throw "Current branch is '$currentBranch'. Switch to '$Branch' or pass -Branch '$currentBranch'."
}

$status = Get-GitOutput status --porcelain
if ($status) {
    throw "Working tree is not clean. Commit or stash local changes before releasing."
}

$versionPath = Join-Path $repoRoot "selfx/version.py"
$versionText = Get-Content -Path $versionPath -Raw -Encoding UTF8
$versionPattern = '__version__\s*=\s*"(?<version>\d+\.\d+\.\d+)"'

if ($versionText -notmatch $versionPattern) {
    throw "Could not find __version__ in $versionPath."
}

$currentVersion = $Matches.version

if (-not $Version) {
    $parts = $currentVersion.Split(".") | ForEach-Object { [int]$_ }

    switch ($Bump) {
        "major" {
            $parts[0] += 1
            $parts[1] = 0
            $parts[2] = 0
        }
        "minor" {
            $parts[1] += 1
            $parts[2] = 0
        }
        "patch" {
            $parts[2] += 1
        }
    }

    $Version = "$($parts[0]).$($parts[1]).$($parts[2])"
}

if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Version must use numeric semantic version format, for example 0.1.35."
}

if ([version]$Version -le [version]$currentVersion) {
    throw "New version '$Version' must be greater than current version '$currentVersion'."
}

$tagName = "v$Version"
$existingTag = Get-GitOutput tag --list $tagName
if ($existingTag) {
    throw "Tag '$tagName' already exists locally."
}

Write-Host "Current version: $currentVersion"
Write-Host "New version:     $Version"
Write-Host "Tag:             $tagName"
Write-Host "Branch:          $Branch"
Write-Host "Remote:          $Remote"

if ($DryRun) {
    Write-Host "Dry run only. No files, commits, tags, or remotes were changed."
    exit 0
}

$newVersionText = [regex]::Replace(
    $versionText,
    $versionPattern,
    "__version__ = `"$Version`"",
    1
)

Set-Content -Path $versionPath -Value $newVersionText -NoNewline -Encoding UTF8

Invoke-Git add selfx/version.py
Invoke-Git commit -m "Bump version to $Version"
Invoke-Git tag $tagName

if ($NoPush) {
    Write-Host "Created local release commit and tag. Push skipped because -NoPush was set."
    Write-Host "To push later, run:"
    Write-Host "  git push $Remote $Branch"
    Write-Host "  git push $Remote $tagName"
    exit 0
}

Invoke-Git push $Remote $Branch
Invoke-Git push $Remote $tagName

Write-Host "Released $tagName."
