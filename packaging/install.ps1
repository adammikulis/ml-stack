# Network installer for ml-stack, in four modes. Re-running any of them upgrades in place.
#
#   irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
#
# `iex` runs the script with no arguments, so a mode is chosen with the environment -- one
# line, and no scriptblock incantation to get a switch past the pipe:
#
#   $env:ML_STACK_MODE="headless"; irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
#   $env:ML_STACK_MODE="dev";      irm ... | iex
#   $env:ML_STACK_MODE="system";   irm ... | iex      (in a PowerShell opened as administrator)
#
# Downloaded to a file it takes switches as well: .\install.ps1 -Headless
#
#   (default)   the app: the release zip for this machine, a window, updates from releases
#   -Headless   a venv under %LOCALAPPDATA%\ml-stack, console scripts on PATH, no window
#   -Dev        a git checkout with an editable install, following main
#   -System     -Headless, per machine: a Scheduled Task at startup, as the user who ran it
#   -Uninstall  takes it off, and leaves the model cache alone
#
# Every step past the install is an ml-stack command, not PowerShell: ml-stack-serve build,
# ml-stack-setup, ml-stack-models fetch, ml-stack-fleet join, ml-stack-doctor.
#
# Unattended: ML_STACK_MODE, ML_STACK_NAME, ML_STACK_PASSPHRASE, ML_STACK_CLUSTER,
# ML_STACK_MODELS, ML_STACK_ADOPT_CACHE, ML_STACK_REF, ML_STACK_OFFLINE_ZIP,
# ML_STACK_OFFLINE_MODELS. Nothing is prompted for when no console is attached.
param(
    [switch]$Headless,
    [switch]$Dev,
    [switch]$System,
    [switch]$Uninstall,
    [switch]$AdoptCache,
    [string]$Models = "",
    [string]$Ref = ""
)
$ErrorActionPreference = "Stop"

$repo    = if ($env:ML_STACK_REPO) { $env:ML_STACK_REPO } else { "adammikulis/ml-stack" }
$api     = "https://api.github.com/repos/$repo/releases/latest"
$gitUrl  = "https://github.com/$repo"
$extras  = "store,hub,web,plot"
$arch    = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x86_64" }
$key     = "ml-stack-windows-$arch"
$offZip  = $env:ML_STACK_OFFLINE_ZIP
$offMod  = $env:ML_STACK_OFFLINE_MODELS
if (-not $Models) { $Models = $env:ML_STACK_MODELS }
if (-not $Ref)    { $Ref    = $env:ML_STACK_REF }

$mode = "app"
if ($env:ML_STACK_MODE) { $mode = $env:ML_STACK_MODE }
if ($Headless) { $mode = "headless" }
if ($Dev)      { $mode = "dev" }
if ($System)   { $mode = "system" }

$script:bin   = ""
$script:track = ""

function Step($what) { Write-Host ""; Write-Host "== $what" }
function Interactive { return [Environment]::UserInteractive -and -not [Console]::IsInputRedirected }

# -- python -------------------------------------------------------------------
# Say how to get one; never install a Python behind somebody's back.
function Find-Python {
    foreach ($name in @("python3.13", "python3.12", "python3.11", "python", "py")) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $found) { continue }
        try {
            & $found.Source -c "import sys; raise SystemExit(sys.version_info < (3, 11))" 2>$null
            if ($LASTEXITCODE -eq 0) { return $found.Source }
        } catch { }
    }
    throw @"
ml-stack needs Python 3.11 or newer. Install it with:
    winget install --id Python.Python.3.13 -e
  (or from https://www.python.org/downloads/windows/), open a new terminal, and run this again.
"@
}

# -- the two firewall rules ---------------------------------------------------
# Windows Defender Firewall blocks the daemon (TCP 8770) and its beacons (UDP 8771)
# inbound by default, so without these two rules the machine is invisible to the rest of
# the fleet. Names and ports match ml_stack.fleet.discovery. One approval prompt.
function Open-Firewall {
    $rules = @(
        @{ Name = "ml-stack traind";    Protocol = "TCP"; Port = 8770 },
        @{ Name = "ml-stack discovery"; Protocol = "UDP"; Port = 8771 }
    )
    $missing = @($rules | Where-Object {
        -not (Get-NetFirewallRule -DisplayName $_.Name -ErrorAction SilentlyContinue) })
    if ($missing.Count -eq 0) { return }
    $lines = ($missing | ForEach-Object {
        "netsh advfirewall firewall add rule name=`"$($_.Name)`" dir=in action=allow " +
        "protocol=$($_.Protocol) localport=$($_.Port)" }) -join " ; "
    Write-Host "Letting the fleet reach this machine (Windows asks for approval once)..."
    try {
        Start-Process powershell -Verb RunAs -Wait -ArgumentList "-NoProfile", "-Command", $lines
    }
    catch {
        Write-Host "Not approved. Other machines will not see this one until, as administrator:"
        Write-Host "  $lines"
    }
}

# -- the app (default) --------------------------------------------------------
function Install-App {
    Step "the app"
    $tmp = Join-Path $env:TEMP ("ml-stack-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp "pkg.zip"
        if ($offZip) {
            Write-Host "installing from $offZip; no network step will run"
            Copy-Item $offZip $zip
        }
        else {
            Write-Host "Looking for the newest ml-stack for Windows $arch..."
            $release = Invoke-RestMethod -Uri $api `
                -Headers @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "ml-stack" }
            $asset = $release.assets | Where-Object { $_.name -like "*$key*" } | Select-Object -First 1
            if (-not $asset) { throw "release $($release.tag_name) has no download for $key" }
            Write-Host "Downloading $($release.tag_name)..."
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
        }
        Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "out") -Force

        $dest = if ($env:ML_STACK_DEST) { $env:ML_STACK_DEST }
                else { Join-Path $env:LOCALAPPDATA "Programs\ml-stack" }
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        Copy-Item -Path (Join-Path $tmp "out\*") -Destination $dest -Recurse -Force
        Add-ToPath $dest
        Open-Firewall
        Write-Host ""
        Write-Host "Installed to $dest"
        Write-Host "Open ml-stack.exe, and type the same passphrase you used on your other machines."
        Write-Host "It downloads gemma-4-E2B on first run (2.6G, about 1.5s a question) and offers"
        Write-Host "the bigger models this machine has room for."
        Start-Process (Join-Path $dest "ml-stack.exe") -ErrorAction SilentlyContinue
    }
    finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

function Add-ToPath($dir) {
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($path -notlike "*$dir*") {
        [Environment]::SetEnvironmentVariable("Path", "$path;$dir", "User")
        Write-Host "Added $dir to your PATH (open a new terminal to pick it up)."
    }
}

# -- headless: a venv and the console scripts ---------------------------------
function Venv-Root {
    if ($env:ML_STACK_PREFIX) { return (Join-Path $env:ML_STACK_PREFIX "venv") }
    if ($mode -eq "system") { return "C:\ProgramData\ml-stack\venv" }
    return (Join-Path $env:LOCALAPPDATA "ml-stack\venv")
}

function New-Venv($venv) {
    $py = Find-Python
    Write-Host "python: $py"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        & $py -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "could not make a virtualenv at $venv" }
    }
    $script:bin = Join-Path $venv "Scripts"
    & (Join-Path $script:bin "python.exe") -m pip install --quiet --upgrade pip
}

function Install-Headless {
    Step "headless"
    New-Venv (Venv-Root)
    $pip = Join-Path $script:bin "pip.exe"
    if ($offZip) {
        Write-Host "installing from $offZip; no network step will run"
        & $pip install --quiet $offZip
    }
    else {
        $want = $Ref
        if (-not $want) {
            try {
                $want = (Invoke-RestMethod -Uri $api `
                    -Headers @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "ml-stack" }).tag_name
            } catch { $want = "main" }
        }
        if (-not $want) { $want = "main" }
        Write-Host "installing ml-stack[$extras] at $want"
        & $pip install --quiet --upgrade "ml-stack[$extras] @ git+$gitUrl@$want"
        if ($LASTEXITCODE -ne 0) { throw "pip could not install ml-stack" }
        # The ref decides how it keeps itself current: a tag follows releases, main follows main.
        if ($want -in @("main", "master")) { $script:track = $want }
    }
    Add-ToPath $script:bin
    Open-Firewall
}

# -- dev: a checkout that follows main ----------------------------------------
function Install-Dev {
    Step "developer"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "this needs git" }
    $src = if ($env:ML_STACK_SRC) { $env:ML_STACK_SRC } else { Join-Path $env:LOCALAPPDATA "ml-stack\src" }
    if (Test-Path (Join-Path $src ".git")) {
        Write-Host "updating $src"
        & git -C $src pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Host "  it has commits main does not; left alone" }
    }
    else {
        Write-Host "cloning into $src"
        New-Item -ItemType Directory -Path (Split-Path $src) -Force | Out-Null
        & git clone $gitUrl $src
        if ($LASTEXITCODE -ne 0) { throw "could not clone $gitUrl" }
    }
    New-Venv (Venv-Root)
    Write-Host "editable install of $src"
    Push-Location $src
    try { & (Join-Path $script:bin "pip.exe") install --quiet -e ".[$extras]" }
    finally { Pop-Location }
    Add-ToPath $script:bin
    Open-Firewall
    $script:track = if ($env:ML_STACK_TRACK) { $env:ML_STACK_TRACK } else { "main" }
}

# -- per machine: at startup, as the user who installed it --------------------
function Install-System {
    Step "per machine"
    $me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw @"
-System installs for the whole machine, so it needs an administrator. Open PowerShell as
administrator and run:
    `$env:ML_STACK_MODE="system"; irm $gitUrl/raw/main/packaging/install.ps1 | iex
"@
    }
    $who = "$env:USERDOMAIN\$env:USERNAME"
    $home_dir = $env:USERPROFILE
    Write-Host "the task will run at startup as $who ($home_dir)"
    # The models already on this disk. Running as the installing user means the service
    # opens that cache where it is -- nothing moved, linked or downloaded twice.
    $cacheArgs = if ($AdoptCache -or $env:ML_STACK_ADOPT_CACHE -eq "yes") { "--adopt" } else { "--same-user" }
    & (Join-Path $script:bin "python.exe") -m ml_stack.fleet.autostart cache `
        --user-cache (Join-Path $home_dir ".cache\huggingface") $cacheArgs
    & (Join-Path $script:bin "python.exe") -m ml_stack.fleet.autostart system `
        --user $who --home $home_dir
    if ($LASTEXITCODE -ne 0) { throw "could not register the startup task" }
}

# -- after any install --------------------------------------------------------
function Build-Llama {
    Step "llama.cpp"
    if ($offZip) { Write-Host "offline: skipping the llama.cpp build"; return }
    $serve = Join-Path $script:bin "ml-stack-serve.exe"
    if (-not (Test-Path $serve)) { Write-Host "skipped: no ml-stack-serve"; return }
    # Most Windows installs have no compiler, so a release build is the default here.
    $from = if ($env:ML_STACK_BUILD -eq "source") { "source" } else { "release" }
    & $serve build --from $from
    if ($LASTEXITCODE -ne 0) { Write-Host "  the build did not finish; 'ml-stack-serve build' retries" }
}

function Show-Sizing {
    Step "what this machine can do"
    $setup = Join-Path $script:bin "ml-stack-setup.exe"
    if (Test-Path $setup) { & $setup } else { Write-Host "skipped" }
}

function Fetch-Models {
    Step "models"
    $want = $Models
    if (-not $want) { $want = if ($mode -eq "app") { "default" } else { "auto" } }
    if ($want -eq "none") { Write-Host "none asked for"; return }
    if ($offMod) { Write-Host "offline: using the models in $offMod; nothing is downloaded"; return }
    $py = Join-Path $script:bin "python.exe"
    $room = & $py -c "from ml_stack.fleet.bench import machine_room; print(machine_room())" 2>$null
    if (-not $room) { $room = 0 }
    $pick = & $py -m ml_stack.fleet.autostart choose --room $room --want $want 2>$null
    if (-not $pick) { Write-Host "no measured model fits this machine; none fetched"; return }
    Write-Host "fetching $pick into the one cache on this machine"
    # ml-stack-models fetch checks every download's sha256 and refuses a mismatch.
    & (Join-Path $script:bin "ml-stack-models.exe") fetch @($pick -split "\s+")
}

function Join-Fleet {
    Step "joining the fleet"
    $fleet = Join-Path $script:bin "ml-stack-fleet.exe"
    if (-not (Test-Path $fleet)) { Write-Host "skipped: no ml-stack-fleet"; return }
    $argv = @("join", "--persist")
    if ($env:ML_STACK_NAME)    { $argv += @("--name", $env:ML_STACK_NAME) }
    if ($env:ML_STACK_CLUSTER) { $argv += @("--group", $env:ML_STACK_CLUSTER) }
    if ($script:track)         { $argv += @("--track", $script:track) }
    if ($env:ML_STACK_PASSPHRASE) { $argv += @("--passphrase", $env:ML_STACK_PASSPHRASE) }
    elseif (-not (Interactive)) {
        Write-Host "no passphrase, and no console to ask at. Set ML_STACK_PASSPHRASE and re-run,"
        Write-Host "or run:  ml-stack-fleet join --persist"
        return
    }
    & $fleet @argv
}

function Check-Over {
    Step "checking it over"
    $doctor = Join-Path $script:bin "ml-stack-doctor.exe"
    if (Test-Path $doctor) { & $doctor } else { Write-Host "skipped" }
}

function Last-Screen {
    Step "done"
    Write-Host ("  machine     " + $(if ($env:ML_STACK_NAME) { $env:ML_STACK_NAME } else { $env:COMPUTERNAME }))
    Write-Host ("  cluster     " + $(if ($env:ML_STACK_CLUSTER) { $env:ML_STACK_CLUSTER } else { "ml-stack" }))
    Write-Host "  open        http://127.0.0.1:8770/ui/"
    $py = Join-Path $script:bin "python.exe"
    if (Test-Path $py) {
        & $py -c "from ml_stack.fleet import updates; s = updates.state(); print('  running     ' + (s['version'] or '?') + '  ' + (s['commit'] or '?'))" 2>$null
    }
    if ($script:track) { Write-Host "  updates     follows $($script:track), whenever nothing is running here" }
    else { Write-Host "  updates     releases, whenever nothing is running here" }
    Write-Host ""
    Write-Host "  next        ml-stack-fleet status    -- who else is in the fleet"
}

function Remove-MlStack {
    Step "removing ml-stack"
    $venv = Venv-Root
    $py = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $py) {
        # uninstall.plan ticks everything ml-stack made for itself and leaves unticked what
        # the person made -- their models and their datasets. Only the ticked ones go.
        $code = @(
            "from pathlib import Path",
            "from ml_stack.fleet import uninstall",
            "root = Path('~/.ml-stack/traind').expanduser()",
            "items = uninstall.plan(root)",
            "went = uninstall.remove(root, [i.key for i in items if i.default])",
            "[print('  removed', n) for n in went.get('removed', [])]",
            "[print('  kept   ', i.name) for i in items if not i.default]"
        ) -join "`n"
        & $py -c $code
    }
    Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "The model cache is left where it is, so coming back downloads nothing again."
    Write-Host "Remove it yourself with:  Remove-Item -Recurse `$HOME\.cache\huggingface\hub"
}

# -- go -----------------------------------------------------------------------
if ($Uninstall) { Remove-MlStack; return }

switch ($mode) {
    "app"      { Install-App }
    "headless" { Install-Headless }
    "dev"      { Install-Dev }
    "system"   { Install-Headless; Install-System }
    default    { throw "unknown mode '$mode' (app, headless, dev, system)" }
}

if ($mode -ne "app") {
    Show-Sizing
    Build-Llama
    Fetch-Models
    Join-Fleet
    Check-Over
    Last-Screen
}
