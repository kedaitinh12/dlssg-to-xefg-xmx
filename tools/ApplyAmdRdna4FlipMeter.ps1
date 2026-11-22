<#
.SYNOPSIS
    Automatically detects AMD RDNA 4+ discrete GPUs (even on multi-GPU / iGPU + dGPU PCs)
    and sets Gmx_DriverFramePacingMaxGenFrames = 5 (up to 6x FG) for OptiScaler Flip Meter.
.PARAMETER Revert
    Removes Gmx_DriverFramePacingMaxGenFrames from detected AMD RDNA 4+ adapter subkeys (restores 2x driver default).
#>
param(
    [switch]$Revert
)

$ErrorActionPreference = 'Stop'

# Auto-elevate to Administrator if needed
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($Revert) { $argList += '-Revert' }
    Write-Host "Requesting Administrator privileges to update AMD RDNA 4+ display registry key..."
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $argList
    exit 0
}

$classPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$valueName = 'Gmx_DriverFramePacingMaxGenFrames'
$targetValue = 5

function Test-IsAmdRdna4OrNewer {
    param([string]$MatchingId, [string]$DriverDesc)

    if ([string]::IsNullOrWhiteSpace($MatchingId)) { return $false }
    if ($MatchingId -notmatch '(?i)VEN_1002.*DEV_([0-9A-F]{4})') { return $false }

    $devId = [Convert]::ToInt32($Matches[1], 16)
    # AMD Navi 4x (RDNA 4: gfx1200/gfx1201, RX 9070 XT / RX 9070 / RX 9060 XT) PCI IDs 0x7550..0x75FF
    if ($devId -ge 0x7550 -and $devId -le 0x75FF) { return $true }

    $desc = if ($DriverDesc) { $DriverDesc.ToUpperInvariant() } else { '' }
    if ($desc -match '780M|880M|890M|8060S|8050S|RADEON\(TM\) GRAPHICS') { return $false }
    return ($desc -match 'RX 90|RADEON AI PRO R9|RDNA 4|RDNA 5|GFX12|GFX13')
}

$matched = @()
Get-ChildItem -Path $classPath -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object {
    $props = Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue
    if ($props -and (Test-IsAmdRdna4OrNewer -MatchingId $props.MatchingDeviceId -DriverDesc $props.DriverDesc)) {
        $matched += [PSCustomObject]@{
            Subkey     = $_.PSChildName
            Path       = $_.PSPath
            DriverDesc = $props.DriverDesc
            MatchId    = $props.MatchingDeviceId
        }
    }
}

if ($matched.Count -eq 0) {
    Write-Warning "No AMD RDNA 4+ GPU detected under $classPath. No registry changes were made."
    exit 1
}

foreach ($gpu in $matched) {
    if ($Revert) {
        Remove-ItemProperty -Path $gpu.Path -Name $valueName -ErrorAction SilentlyContinue
        Write-Host "[REVERTED] $($gpu.Subkey) ($($gpu.DriverDesc)) -> removed $valueName (restored 2x default)" -ForegroundColor Yellow
    } else {
        New-ItemProperty -Path $gpu.Path -Name $valueName -Value $targetValue -PropertyType DWord -Force | Out-Null
        Write-Host "[APPLIED]  $($gpu.Subkey) ($($gpu.DriverDesc)) -> $valueName = $targetValue (up to 6x Flip Meter)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Done! Restart Windows for the AMD display driver to apply the change." -ForegroundColor Cyan
