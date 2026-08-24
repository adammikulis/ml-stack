# Network installer for ml-stack. Downloads the current release for this machine.
#
#   irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
#
# Everything it fetches comes from the GitHub release for the repository below.
$ErrorActionPreference = "Stop"

$repo = if ($env:ML_STACK_REPO) { $env:ML_STACK_REPO } else { "adammikulis/ml-stack" }
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x86_64" }
$key  = "ml-stack-windows-$arch"

Write-Host "Looking for the newest ml-stack for Windows $arch..."
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" `
    -Headers @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "ml-stack" }

$asset = $release.assets | Where-Object { $_.name -like "*$key*" } | Select-Object -First 1
if (-not $asset) {
    throw "release $($release.tag_name) has no download for $key"
}

$tmp = Join-Path $env:TEMP ("ml-stack-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Downloading $($release.tag_name)..."
    $zip = Join-Path $tmp "pkg.zip"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "out") -Force

    $dest = if ($env:ML_STACK_DEST) { $env:ML_STACK_DEST }
            else { Join-Path $env:LOCALAPPDATA "Programs\ml-stack" }
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Copy-Item -Path (Join-Path $tmp "out\*") -Destination $dest -Recurse -Force

    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($path -notlike "*$dest*") {
        [Environment]::SetEnvironmentVariable("Path", "$path;$dest", "User")
        Write-Host "Added $dest to your PATH (open a new terminal to pick it up)."
    }
    Write-Host ""
    Write-Host "Installed to $dest"
    Write-Host "Open ml-stack.exe, and type the same passphrase you used on your other machines."
    Start-Process (Join-Path $dest "ml-stack.exe") -ErrorAction SilentlyContinue
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
