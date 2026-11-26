# SPDX-License-Identifier: MIT
#
# Compiles the 16 unique D3D12 WMMA / dot4add_i8packed compute shaders (and 17 <program>.dxbc files)
# from src\MfgPort\Wmma\xefg_wmma_layers.hlsl using dxc.exe (-T cs_6_6 -O3) into <Output>\wmma\<program>.dxbc.
#
# Usage:
#   ./tools/GenerateXeFgWmma.ps1 -Output bin\Release\xefg_mfg_programs\wmma

param(
    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$Source = "",

    [string]$DxcPath = ""
)

$ErrorActionPreference = "Stop"

if (-not $Source) {
    $Source = Join-Path $PSScriptRoot "..\src\MfgPort\Wmma\xefg_wmma_layers.hlsl"
}

if (-not (Test-Path $Source)) {
    throw "WMMA HLSL shader source not found: $Source"
}

function Find-Dxc {
    param([string]$ExplicitPath)
    if ($ExplicitPath -and (Test-Path $ExplicitPath)) {
        return (Resolve-Path $ExplicitPath).Path
    }
    $cmd = Get-Command dxc.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $kitsRoot = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path $kitsRoot) {
        $found = Get-ChildItem -Path $kitsRoot -Recurse -Filter "dxc.exe" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\dxc\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    if ($env:VULKAN_SDK) {
        $vkDxc = Join-Path $env:VULKAN_SDK "Bin\dxc.exe"
        if (Test-Path $vkDxc) { return $vkDxc }
    }
    return $null
}

$dxc = Find-Dxc -ExplicitPath $DxcPath
if (-not $dxc) {
    throw "dxc.exe not found in PATH, Windows Kits, or VULKAN_SDK."
}

New-Item -ItemType Directory -Path $Output -Force | Out-Null

# Fast-path no-op when all 17 <program>.dxbc files and wmma_layers.tsv are already newer than the HLSL source
$manifestPath = Join-Path $Output "wmma_layers.tsv"
$splitOutDir = Join-Path $Output "dot4_split"
$splitManifestPath = Join-Path $splitOutDir "wmma_layers.tsv"
$srcTime = (Get-Item $Source).LastWriteTimeUtc
$expectedProgs = @("aa016078","198d6bf8","dcd8becf","efbe3788","50f0c205","a3c3caa1","9481d452","06e97933","3ed03daf","748b0baf","cb391582","fc1b8a04","3459485","3452733","55e5d6b9","34517926","3453044")
if ((Test-Path $manifestPath) -and (Test-Path $splitManifestPath) -and
    ((Get-Item $manifestPath).LastWriteTimeUtc -ge $srcTime) -and
    ((Get-Item $splitManifestPath).LastWriteTimeUtc -ge $srcTime)) {
    $allCurrent = $true
    foreach ($dirCheck in @($Output, $splitOutDir)) {
        foreach ($prog in $expectedProgs) {
            $pf = Join-Path $dirCheck "$prog.dxbc"
            if (-not (Test-Path $pf) -or ((Get-Item $pf).LastWriteTimeUtc -lt $srcTime)) {
                $allCurrent = $false
                break
            }
        }
        if (-not $allCurrent) { break }
    }
    if ($allCurrent) {
        Write-Host "D3D12 WMMA/dot4 shaders in $Output (and dot4_split) are up to date; skipping DXC compilation."
        exit 0
    }
}

$kernels = @(
    @{ Name="L_aa016078";     Family=0; In=1;  Out=1;  Bias=1; Pool=1; Relu=1; TileH=16; ZSplit=1; AccBits=16; Nt=4; Programs=@("aa016078") },
    @{ Name="L_198d6bf8";     Family=0; In=1;  Out=2;  Bias=0; Pool=1; Relu=1; TileH=8;  ZSplit=1; AccBits=16; Nt=4; Programs=@("198d6bf8") },
    @{ Name="L_dcd8becf";     Family=0; In=2;  Out=4;  Bias=0; Pool=1; Relu=1; TileH=16; ZSplit=2; AccBits=16; Nt=4; Programs=@("dcd8becf") },
    @{ Name="L_efbe3788";     Family=0; In=4;  Out=8;  Bias=0; Pool=1; Relu=1; TileH=8;  ZSplit=4; AccBits=16; Nt=4; Programs=@("efbe3788") },
    @{ Name="L_50f0c205";     Family=0; In=8;  Out=4;  Bias=0; Pool=0; Relu=1; TileH=4;  ZSplit=4; AccBits=16; Nt=4; Programs=@("50f0c205") },
    @{ Name="L_a3c3caa1";     Family=0; In=4;  Out=2;  Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=1; AccBits=16; Nt=4; Programs=@("a3c3caa1") },
    @{ Name="L_9481d452";     Family=0; In=1;  Out=1;  Bias=0; Pool=0; Relu=0; TileH=16; ZSplit=1; AccBits=16; Nt=4; Programs=@("9481d452") },
    @{ Name="L_06e97933";     Family=0; In=2;  Out=1;  Bias=0; Pool=0; Relu=0; TileH=16; ZSplit=1; AccBits=16; Nt=4; Programs=@("06e97933") },
    @{ Name="L_3ed03daf";     Family=1; In=1;  Out=1;  Bias=0; Pool=0; Relu=1; TileH=16; ZSplit=1; AccBits=16; Nt=4; Programs=@("3ed03daf") },
    @{ Name="L_748b0baf";     Family=1; In=4;  Out=4;  Bias=0; Pool=0; Relu=1; TileH=8;  ZSplit=1; AccBits=16; Nt=4; Programs=@("748b0baf") },
    @{ Name="L_cb391582";     Family=1; In=2;  Out=2;  Bias=0; Pool=0; Relu=1; TileH=16; ZSplit=1; AccBits=16; Nt=4; Programs=@("cb391582") },
    @{ Name="L_fc1b8a04";     Family=2; In=14; Out=8;  Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=7; AccBits=16; Nt=4; Programs=@("fc1b8a04") },
    @{ Name="L_3459485";      Family=2; In=14; Out=8;  Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=7; AccBits=18; Nt=4; Programs=@("3459485") },
    @{ Name="L_3452733";      Family=2; In=14; Out=8;  Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=7; AccBits=19; Nt=4; Programs=@("3452733") },
    @{ Name="L_flow16_acc17"; Family=2; In=4;  Out=16; Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=2; AccBits=17; Nt=4; Programs=@("55e5d6b9", "34517926") },
    @{ Name="L_3453044";      Family=2; In=4;  Out=16; Bias=0; Pool=0; Relu=0; TileH=8;  ZSplit=2; AccBits=18; Nt=4; Programs=@("3453044") }
)

function Build-Dot4Set {
    param([string]$OutDir, [string]$Label, [string[]]$ExtraDefines)
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $tsvLines = @("program`tkernel`tfamily`tin`tout`ttile_h`tgrid_z`tbytes`tsha256")
    foreach ($k in $kernels) {
        $kernelOut = Join-Path $OutDir "$($k.Name).dxbc"
        $argsList = @(
            "-T", "cs_6_6",
            "-E", "main",
            "-O3",
            "-DLAYER_FAMILY=$($k.Family)",
            "-DLAYER_IN=$($k.In)",
            "-DLAYER_OUT=$($k.Out)",
            "-DLAYER_BIAS=$($k.Bias)",
            "-DLAYER_POOL=$($k.Pool)",
            "-DLAYER_RELU=$($k.Relu)",
            "-DLAYER_TILE_H=$($k.TileH)",
            "-DLAYER_ZSPLIT=$($k.ZSplit)",
            "-DLAYER_ACC_BITS=$($k.AccBits)",
            "-DLAYER_NT=$($k.Nt)"
        ) + $ExtraDefines + @("-Fo", $kernelOut, $Source)

        & $dxc @argsList
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $kernelOut)) {
            throw "dxc.exe failed compiling [$Label] $($k.Name) (exit code $LASTEXITCODE)"
        }

        $bytes = (Get-Item $kernelOut).Length
        $hash = (Get-FileHash -Path $kernelOut -Algorithm SHA256).Hash.ToLowerInvariant()

        foreach ($prog in $k.Programs) {
            $progOut = Join-Path $OutDir "$prog.dxbc"
            Copy-Item -Path $kernelOut -Destination $progOut -Force
            $tsvLines += "$prog`t$($k.Name)`t$($k.Family)`t$($k.In)`t$($k.Out)`t$($k.TileH)`t$($k.ZSplit)`t$bytes`t$hash"
        }
        Write-Host ("ok   [{0}] {1,-18} ({2} bytes, sha256={3}...)" -f $Label, "$($k.Name).dxbc", $bytes, $hash.Substring(0, 12))
    }
    Set-Content -Path (Join-Path $OutDir "wmma_layers.tsv") -Value ($tsvLines -join "`n") -Encoding UTF8
    Write-Host "Built 16 unique D3D12 dot4 kernels (17 layer overrides) [$Label] into $OutDir"
}

Build-Dot4Set -OutDir $Output -Label "dot4-acc" -ExtraDefines @()
Build-Dot4Set -OutDir (Join-Path $Output "dot4_split") -Label "dot4-split" -ExtraDefines @("-DDOT4_SPLIT=1")

# Also build byte-identical motion-aware splat DXIL replacements (splat.1 / splat.2) into <Output>\..\replacements
$splatSource = Join-Path $PSScriptRoot "..\src\MfgPort\splat_motion_aware.hlsl"
if (Test-Path $splatSource) {
    $replOut = Join-Path (Split-Path -Path $Output -Parent) "replacements"
    New-Item -ItemType Directory -Path $replOut -Force | Out-Null
    $splatVariants = @(
        @{ Name = "splat_motion_aware";          Defines = @() },
        @{ Name = "splat_motion_aware_inverted"; Defines = @("-DSPLAT_INVERTED_DEPTH=1") }
    )
    foreach ($sv in $splatVariants) {
        $dxilOut = Join-Path $replOut "$($sv.Name).dxil"
        $splatArgs = @(
            "-nologo",
            "-T", "cs_6_0",
            "-E", "main",
            "-O3",
            "-Qstrip_reflect",
            "-Qstrip_debug"
        ) + $sv.Defines + @("-Fo", $dxilOut, $splatSource)

        & $dxc @splatArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dxilOut)) {
            throw "dxc.exe failed compiling $($sv.Name) (exit code $LASTEXITCODE)"
        }
        $sBytes = (Get-Item $dxilOut).Length
        $sHash = (Get-FileHash -Path $dxilOut -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host ("ok   {0,-28} ({1} bytes, sha256={2}...)" -f "$($sv.Name).dxil", $sBytes, $sHash.Substring(0, 12))
    }
}
