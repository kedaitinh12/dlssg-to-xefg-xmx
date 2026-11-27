param([Parameter(Mandatory = $true)][string]$Output)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root 'third_party/glslang/glslang.exe'
$pin = Join-Path $root 'third_party/glslang/SHA256.txt'
if (-not (Test-Path $exe -PathType Leaf)) { throw "Required compiler missing: $exe" }
$expected = ((Get-Content $pin -Raw).Trim() -split '\s+')[0]
if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid glslang SHA-256 pin' }
if ((Get-FileHash $exe -Algorithm SHA256).Hash -ne $expected) { throw 'glslang SHA-256 mismatch' }
& $exe --version
if ($LASTEXITCODE -ne 0) { throw 'Vendored glslang cannot run' }
python (Join-Path $PSScriptRoot 'GenerateXeFgCoopMat.py') --glslang $exe --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Cooperative-matrix shader compilation failed' }
python (Join-Path $PSScriptRoot 'CheckXeFgCoopMat.py') $Output
if ($LASTEXITCODE -ne 0) { throw 'Cooperative-matrix shader validation failed' }

# Also compile Vulkan SPIR-V motion-aware splat replacements (splat.1 / splat.2) into <xefg_mfg_programs>\replacements
$splatComp = Join-Path $root 'src/vulkan/shaders/xefg_splat_motion_aware.comp'
if (Test-Path $splatComp -PathType Leaf) {
    $programsRoot = Split-Path (Split-Path $Output -Parent) -Parent
    $replOut = Join-Path $programsRoot 'replacements'
    New-Item -ItemType Directory -Path $replOut -Force | Out-Null
    $splatVariants = @(
        @{ Name = 'splat_motion_aware';          Defines = @() },
        @{ Name = 'splat_motion_aware_inverted'; Defines = @('-DSPLAT_INVERTED_DEPTH=1') }
    )
    foreach ($sv in $splatVariants) {
        $spvOut = Join-Path $replOut "$($sv.Name).spv"
        $glslArgs = @('-V', '--target-env', 'vulkan1.3') + $sv.Defines + @('-o', $spvOut, $splatComp)
        & $exe @glslArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $spvOut -PathType Leaf)) {
            throw "Vulkan motion-aware splat shader compilation failed for $($sv.Name)"
        }
        $spvBytes = [System.IO.File]::ReadAllBytes($spvOut)
        if ($spvBytes.Length -lt 20 -or ($spvBytes.Length % 4) -ne 0 -or [System.BitConverter]::ToUInt32($spvBytes, 0) -ne 0x07230203) {
            throw "Invalid SPIR-V output for $($sv.Name): $($spvBytes.Length) bytes"
        }
        $spvHash = (Get-FileHash -Path $spvOut -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host ("ok   {0,-32} ({1} bytes, sha256={2}...)" -f "$($sv.Name).spv", $spvBytes.Length, $spvHash.Substring(0, 12))
    }
}
