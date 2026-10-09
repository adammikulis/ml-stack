# Network installer for poolhouse, in four modes. Re-running any of them upgrades in place.
#
#   irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
#
# `iex` runs the script with no arguments, so a mode is chosen with the environment -- one
# line, and no scriptblock incantation to get a switch past the pipe:
#
#   $env:POOLHOUSE_MODE="headless"; irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
#   $env:POOLHOUSE_MODE="dev";      irm ... | iex
#   $env:POOLHOUSE_MODE="system";   irm ... | iex      (in a PowerShell opened as administrator)
#
# Downloaded to a file it takes switches as well: .\install.ps1 -Headless
#
#   (default)   the app: the release zip for this machine, a window, updates from releases
#   -Headless   a venv under %LOCALAPPDATA%\poolhouse, console scripts on PATH, no window
#   -Dev        a git checkout with an immutable install, following 0.3dev
#   -System     -Headless, per machine: a Scheduled Task at startup, as the user who ran it
#   -Uninstall  takes it off, and leaves the model cache alone
#
# Every step past the install is a poolhouse command, not PowerShell: poolhouse-serve build,
# poolhouse-setup, poolhouse-models fetch, poolhouse-cluster join, poolhouse-doctor.
#
# Unattended: POOLHOUSE_MODE, POOLHOUSE_NAME, POOLHOUSE_PASSPHRASE, POOLHOUSE_CLUSTER,
# POOLHOUSE_MODELS, POOLHOUSE_ADOPT_CACHE, POOLHOUSE_REF, POOLHOUSE_OFFLINE_ZIP,
# POOLHOUSE_OFFLINE_MODELS. Nothing is prompted for when no console is attached.
#
# POOLHOUSE_OFFLINE_WHEELS=C:\dir names the wheels the extras are installed from offline; a
# `wheels` directory beside the zip is used without being named, and
# `python packaging\build.py --wheelhouse` fills one.
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

$repo    = if ($env:POOLHOUSE_REPO) { $env:POOLHOUSE_REPO } else { "adammikulis/ml-stack" }
$api     = "https://api.github.com/repos/$repo/releases/latest"
$gitUrl  = "https://github.com/$repo"
$python  = "3.13"
$extras  = "store,hub,web,plot,graph,coordinator,agents"
$arch    = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x86_64" }
$key     = "poolhouse-windows-$arch"
$offZip  = $env:POOLHOUSE_OFFLINE_ZIP
$offMod  = $env:POOLHOUSE_OFFLINE_MODELS
$offWhl  = $env:POOLHOUSE_OFFLINE_WHEELS
if (-not $Models) { $Models = $env:POOLHOUSE_MODELS }
if (-not $Ref)    { $Ref    = $env:POOLHOUSE_REF }

$mode = "app"
if ($env:POOLHOUSE_MODE) { $mode = $env:POOLHOUSE_MODE }
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
    # The launcher is how a version is named on Windows; -c prints the interpreter behind it.
    $launcher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $exe = & $launcher.Source "-$python" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch { }
    }
    $direct = Get-Command "python$python" -ErrorAction SilentlyContinue
    if ($direct) { return $direct.Source }
    throw @"
poolhouse runs on Python $python. Install it with:
    winget install --id Python.Python.3.13 -e
  then open a new terminal and run this again.
"@
}

# -- the two firewall rules ---------------------------------------------------
# Windows Defender Firewall blocks the daemon (TCP 8770) and its beacons (UDP 8771)
# inbound by default, so without these two rules the machine is invisible to the rest of
# the fleet. Names and ports match poolhouse.fleet.discovery. One approval prompt.
function Open-Firewall {
    $rules = @(
        @{ Name = "poolhouse traind";    Protocol = "TCP"; Port = 8770 },
        @{ Name = "poolhouse discovery"; Protocol = "UDP"; Port = 8771 }
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
    $tmp = Join-Path $env:TEMP ("poolhouse-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp "pkg.zip"
        if ($offZip) {
            Write-Host "installing from $offZip; no network step will run"
            Copy-Item $offZip $zip
        }
        else {
            Write-Host "Looking for the newest poolhouse for Windows $arch..."
            $release = Invoke-RestMethod -Uri $api `
                -Headers @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "poolhouse" }
            $asset = $release.assets | Where-Object { $_.name -like "*$key*" } | Select-Object -First 1
            if (-not $asset) { throw "release $($release.tag_name) has no download for $key" }
            Write-Host "Downloading $($release.tag_name)..."
            $want = ([string]$asset.digest) -replace '^sha256:', ''
            if ($want -notmatch '^[0-9a-fA-F]{64}$') {
                throw "release $($release.tag_name) reports no sha256 for $key, so it cannot be checked"
            }
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
            if ((Get-FileHash -Algorithm SHA256 -Path $zip).Hash -ne $want.ToUpper()) {
                throw "the download does not match the sha256 GitHub reports for it"
            }
        }
        Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "out") -Force

        $dest = if ($env:POOLHOUSE_DEST) { $env:POOLHOUSE_DEST }
                else { Join-Path $env:LOCALAPPDATA "Programs\poolhouse" }
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        # The window is a Windows installer; the daemon beside it is copied as it is.
        $setup = Get-ChildItem -Path (Join-Path $tmp "out") -Filter "*-setup.exe" -Recurse |
                 Select-Object -First 1
        if ($setup) {
            Start-Process -FilePath $setup.FullName -ArgumentList "/S" -Wait
            Get-ChildItem -Path (Join-Path $tmp "out") -Filter "poolhouse-headless*" -Recurse |
                ForEach-Object { Copy-Item $_.FullName -Destination $dest -Force }
        }
        else { Copy-Item -Path (Join-Path $tmp "out\*") -Destination $dest -Recurse -Force }
        Add-ToPath $dest
        Open-Firewall
        Write-Host ""
        Write-Host "Installed to $dest"
        Write-Host "Open poolhouse to name this device and choose Dev or Prod."
        Write-Host "Setup downloads continue in the background while you finish onboarding."
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
    if ($env:POOLHOUSE_PREFIX) { return (Join-Path $env:POOLHOUSE_PREFIX "venv") }
    if ($mode -eq "system") { return "C:\ProgramData\poolhouse\venv" }
    return (Join-Path $env:LOCALAPPDATA "poolhouse\venv")
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

function Find-Wheelhouse($zip) {
    if ($offWhl) { return $offWhl }
    $beside = Join-Path (Split-Path -Parent $zip) "wheels"
    if (Test-Path $beside) { return $beside }
    return ""
}

# pip takes a direct reference as a URL, and file:// wants forward slashes and three of
# them: file:///C:/dir/pkg.whl.
function Local-Uri($path) {
    $slashed = $path -replace "\\", "/"
    if (-not $slashed.StartsWith("/")) { $slashed = "/$slashed" }
    return "file://$slashed"
}

# The extras come from wheels on the disk. Without them, poolhouse and nothing else.
function Install-Offline($pip) {
    $abs = (Resolve-Path $offZip).Path
    $house = Find-Wheelhouse $abs
    if ($house) {
        Write-Host "extras from the wheels in $house"
        try {
            & $pip install --quiet --no-index --find-links $house `
                "poolhouse[$extras] @ $(Local-Uri $abs)"
            if ($LASTEXITCODE -eq 0) { return }
        } catch { }
        Write-Host "  $house does not hold every wheel poolhouse[$extras] needs"
    }
    & $pip install --quiet $abs
    if ($LASTEXITCODE -ne 0) { throw "could not install $offZip" }
}

function Install-Headless {
    Step "headless"
    New-Venv (Venv-Root)
    $pip = Join-Path $script:bin "pip.exe"
    if ($offZip) {
        Write-Host "installing from $offZip; no network step will run"
        Install-Offline $pip
    }
    else {
        $want = $Ref
        if (-not $want) {
            try {
                $want = (Invoke-RestMethod -Uri $api `
                    -Headers @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "poolhouse" }).tag_name
            } catch { $want = "main" }
        }
        if (-not $want) { $want = "main" }
        Write-Host "installing poolhouse[$extras] at $want"
        & $pip install --quiet --upgrade "poolhouse[$extras] @ git+$gitUrl@$want"
        if ($LASTEXITCODE -ne 0) { throw "pip could not install poolhouse" }
        # The ref decides how it keeps itself current: a tag follows releases, main follows main.
        if ($want -in @("main", "master")) { $script:track = $want }
    }
    Add-ToPath $script:bin
    Open-Firewall
}

# -- dev: a checkout that follows development ----------------------------------------
function Install-Dev {
    Step "developer"
    $script:track = if ($env:POOLHOUSE_TRACK) { $env:POOLHOUSE_TRACK } else { "0.3dev" }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "this needs git" }
    $src = if ($env:POOLHOUSE_SRC) { $env:POOLHOUSE_SRC } else { Join-Path $env:LOCALAPPDATA "poolhouse\src" }
    if (Test-Path (Join-Path $src ".git")) {
        $branch = & git -C $src branch --show-current
        if ($branch -ne $script:track) { throw "$src must be on $script:track; choose a separate POOLHOUSE_SRC" }
        Write-Host "updating $src"
        & git -C $src pull --ff-only
        if ($LASTEXITCODE -ne 0) { throw "could not fast-forward $src" }
    }
    else {
        Write-Host "cloning into $src"
        New-Item -ItemType Directory -Path (Split-Path $src) -Force | Out-Null
        & git clone --branch $script:track $gitUrl $src
        if ($LASTEXITCODE -ne 0) { throw "could not clone $gitUrl" }
    }
    New-Venv (Venv-Root)
    Write-Host "immutable install of $src"
    Push-Location $src
    try { & (Join-Path $script:bin "pip.exe") install --quiet ".[$extras]"; if ($LASTEXITCODE -ne 0) { throw "could not install $src" } }
    finally { Pop-Location }
    & (Join-Path $script:bin "python.exe") -c 'import sys; from pathlib import Path; from poolhouse.fleet.runtime_wheel import install_checkout; code, note = install_checkout(Path(sys.argv[1]), timeout=1800); print(note); raise SystemExit(code)' $src
    if ($LASTEXITCODE -ne 0) { throw "could not install the committed runtime from $src" }
    Add-ToPath $script:bin
    Open-Firewall
}

# -- per machine: at startup, as the user who installed it --------------------
function Install-System {
    Step "per machine"
    $me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw @"
-System installs for the whole machine, so it needs an administrator. Open PowerShell as
administrator and run:
    `$env:POOLHOUSE_MODE="system"; irm $gitUrl/raw/main/packaging/install.ps1 | iex
"@
    }
    $who = "$env:USERDOMAIN\$env:USERNAME"
    $home_dir = $env:USERPROFILE
    Write-Host "the task will run at startup as $who ($home_dir)"
    # The models already on this disk. Running as the installing user means the service
    # opens that cache where it is -- nothing moved, linked or downloaded twice.
    $cacheArgs = if ($AdoptCache -or $env:POOLHOUSE_ADOPT_CACHE -eq "yes") { "--adopt" } else { "--same-user" }
    & (Join-Path $script:bin "python.exe") -m poolhouse.fleet.autostart cache `
        --user-cache (Join-Path $home_dir ".cache\huggingface") $cacheArgs
    & (Join-Path $script:bin "python.exe") -m poolhouse.fleet.autostart system `
        --user $who --home $home_dir
    if ($LASTEXITCODE -ne 0) { throw "could not register the startup task" }
}

# -- after any install --------------------------------------------------------
function Build-Llama {
    Step "llama.cpp"
    if ($offZip) { Write-Host "offline: skipping the llama.cpp build"; return }
    $serve = Join-Path $script:bin "poolhouse-serve.exe"
    if (-not (Test-Path $serve)) { Write-Host "skipped: no poolhouse-serve"; return }
    # Most Windows installs have no compiler, so a release build is the default here.
    $from = if ($env:POOLHOUSE_BUILD -eq "source") { "source" } else { "release" }
    & $serve build --from $from
    if ($LASTEXITCODE -ne 0) { Write-Host "  the build did not finish; 'poolhouse-serve build' retries" }
}

function Show-Sizing {
    Step "what this machine can do"
    $setup = Join-Path $script:bin "poolhouse-setup.exe"
    if (Test-Path $setup) { & $setup } else { Write-Host "skipped" }
}

function Fetch-Models {
    Step "models"
    $want = $Models
    if (-not $want) { $want = if ($mode -eq "app") { "default" } else { "auto" } }
    if ($want -eq "none") { Write-Host "none asked for"; return }
    if ($offMod) { Write-Host "offline: using the models in $offMod; nothing is downloaded"; return }
    $py = Join-Path $script:bin "python.exe"
    $room = & $py -c "from poolhouse.hub import machine_room; print(machine_room())" 2>$null
    if (-not $room) { $room = 0 }
    $pick = & $py -m poolhouse.fleet.autostart choose --room $room --want $want 2>$null
    if (-not $pick) { Write-Host "no measured model fits this machine; none fetched"; return }
    Write-Host "fetching $pick into the one cache on this machine"
    # poolhouse-models fetch checks every download's sha256 and refuses a mismatch.
    & (Join-Path $script:bin "poolhouse-models.exe") fetch @($pick -split "\s+")
}

function Join-Fleet {
    Step "joining the fleet"
    $fleet = Join-Path $script:bin "poolhouse-cluster.exe"
    if (-not (Test-Path $fleet)) { Write-Host "skipped: no poolhouse-cluster"; return }
    $argv = @("join", "--persist")
    if ($env:POOLHOUSE_NAME)    { $argv += @("--name", $env:POOLHOUSE_NAME) }
    if ($env:POOLHOUSE_CLUSTER) { $argv += @("--group", $env:POOLHOUSE_CLUSTER) }
    if ($script:track)         { $argv += @("--track", $script:track) }
    if ($env:POOLHOUSE_PASSPHRASE) { $argv += @("--passphrase", $env:POOLHOUSE_PASSPHRASE) }
    elseif (-not (Interactive)) {
        Write-Host "no passphrase, and no console to ask at. Set POOLHOUSE_PASSPHRASE and re-run,"
        Write-Host "or run:  poolhouse-cluster join --persist"
        return
    }
    & $fleet @argv
}

function Show-WhatCameWithIt {
    Step "what came with it"
    $py = Join-Path $script:bin "python.exe"
    if (-not (Test-Path $py)) { Write-Host "skipped"; return }
    & $py -m poolhouse.installed
    if ($LASTEXITCODE -ne 0 -and $offZip) {
        Write-Host "    on a machine with no network these come from wheels on its disk:"
        Write-Host "    put them in one directory and name it with POOLHOUSE_OFFLINE_WHEELS=C:\dir"
    }
}

function Check-Over {
    Step "checking it over"
    $doctor = Join-Path $script:bin "poolhouse-doctor.exe"
    if (Test-Path $doctor) { & $doctor } else { Write-Host "skipped" }
}

function Last-Screen {
    Step "done"
    $py = Join-Path $script:bin "python.exe"
    if (-not (Test-Path $py)) { Write-Host "skipped"; return }
    $name = $(if ($env:POOLHOUSE_NAME) { $env:POOLHOUSE_NAME } else { $env:COMPUTERNAME })
    & $py -m poolhouse.fleet.autostart done --name $name --track "$($script:track)"
}

function Remove-Poolhouse {
    Step "removing poolhouse"
    $venv = Venv-Root
    $py = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $py) {
        # uninstall.plan ticks everything poolhouse made for itself and leaves unticked what
        # the person made -- their models and their datasets. Only the ticked ones go.
        $code = @(
            "from poolhouse.fleet import uninstall",
            "from poolhouse.home import state",
            "root = state('traind')",
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
if ($Uninstall) { Remove-Poolhouse; return }

switch ($mode) {
    "app"      { Install-App }
    "headless" { Install-Headless }
    "dev"      { Install-Dev }
    "system"   { Install-Headless; Install-System }
    default    { throw "unknown mode '$mode' (app, headless, dev, system)" }
}

if ($mode -ne "app") {
    Build-Llama
    Fetch-Models
    Join-Fleet
    Show-Sizing
    Show-WhatCameWithIt
    Check-Over
    Last-Screen
}
