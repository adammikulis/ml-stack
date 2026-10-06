param(
    [Parameter(Mandatory=$true)][string]$Sdk,
    [Parameter(Mandatory=$true)][string]$Gradle,
    [Parameter(Mandatory=$true)][string]$AndroidPlugin,
    [Parameter(Mandatory=$true)][int]$AndroidApi,
    [Parameter(Mandatory=$true)][string]$SigningStore,
    [Parameter(Mandatory=$true)][string]$SigningPassword,
    [Parameter(Mandatory=$true)][string]$Scratch
)
$ErrorActionPreference = 'Stop'
$env:ANDROID_HOME = (Resolve-Path -LiteralPath $Sdk).Path
$env:GRADLE_USER_HOME = Join-Path $Scratch 'gradle-cache'
$env:ANDROID_SIGNING_STORE = (Resolve-Path -LiteralPath $SigningStore).Path
$env:ANDROID_SIGNING_PASSWORD = $SigningPassword
Push-Location $PSScriptRoot
try {
    & $Gradle --no-daemon --max-workers=2 "-PandroidPlugin=$AndroidPlugin" "-PandroidApi=$AndroidApi" protocolChecks lintRelease assembleRelease
    if ($LASTEXITCODE -ne 0) { throw 'Android build failed.' }
    Get-Item -LiteralPath (Join-Path $PSScriptRoot 'build/outputs/apk/release/ml-stack-android-release.apk')
} finally {
    Remove-Item Env:ANDROID_SIGNING_PASSWORD -ErrorAction SilentlyContinue
    Pop-Location
}
