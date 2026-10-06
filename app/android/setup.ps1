param(
    [Parameter(Mandatory=$true)][string]$Scratch,
    [Parameter(Mandatory=$true)][int]$AndroidApi,
    [Parameter(Mandatory=$true)][string]$BuildTools,
    [switch]$SdkDownloadLicenseAccepted
)
$ErrorActionPreference = 'Stop'
if (-not $SdkDownloadLicenseAccepted) {
    throw 'A person must accept https://developer.android.com/studio#downloads before downloading the Android SDK.'
}
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
$archive = Join-Path $Scratch 'android-command-line-tools.zip'
Invoke-WebRequest -UseBasicParsing -Uri 'https://dl.google.com/android/repository/commandlinetools-win-15859902_latest.zip' -OutFile $archive
$expected = '90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Android command-line tools checksum mismatch.'
}
$sdk = Join-Path $Scratch 'android-sdk'
$extracted = Join-Path $Scratch 'android-tools-extracted'
if (Test-Path -LiteralPath $extracted) { throw 'Choose a fresh scratch directory for Android setup.' }
Expand-Archive -LiteralPath $archive -DestinationPath $extracted
New-Item -ItemType Directory -Force -Path (Join-Path $sdk 'cmdline-tools') | Out-Null
Move-Item -LiteralPath (Join-Path $extracted 'cmdline-tools') -Destination (Join-Path $sdk 'cmdline-tools/latest')
$manager = Join-Path $sdk 'cmdline-tools/latest/bin/sdkmanager.bat'
& $manager "--sdk_root=$sdk" 'platform-tools' "platforms;android-$AndroidApi" "build-tools;$BuildTools"
if ($LASTEXITCODE -ne 0) { throw 'Android SDK package installation failed.' }
Write-Output $sdk
