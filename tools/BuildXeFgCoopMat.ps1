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
