# dlssg-to-xefg-xmx

Replaces DLSS Frame Generation with Intel XeSS Frame Generation (XeFG), up to 6x. Inspired by Nukem9's
[dlssg-to-fsr3](https://github.com/Nukem9/dlssg-to-fsr3).

`libxess_fg.dll` only exports XeFG's proxy swap chain. Internally that swap chain drives an unexported
`xefgD3D12CreateContext / Init / Execute` API, which this dll locates and calls directly with the DLSSG inputs. That's
what makes it work as an `nvngx_dlssg` replacement, and it lifts XeFG's 4x cap and its non-Intel 2x clamp without
patching any code.

DX12 games use XeFG directly. XeFG has no Vulkan API, so for Vulkan games the dll runs it on the game's `VkDevice` through
MFG-Unlocker's D3D12-on-Vulkan layer, which is built into it. `MFGUnlocker.asi` isn't needed, see [Vulkan](#vulkan).

## Usage

`dlssg_to_xefg_intel_is_better.dll` exports the same `NVSDK_NGX_D3D12_*` functions as `nvngx_dlssg.dll`, for a loader
that calls those directly. Put `libxess_fg.dll` (XeFG 1.3.1 or 1.3.4) next to it, unless the game loads a compatible one
itself.

It exports the `NVSDK_NGX_VULKAN_*` functions of `nvngx_dlssg.dll` as well and runs Vulkan games itself. The Vulkan models it
runs go next to the dll, in `xefg_mfg_programs\vulkan`, see [Vulkan](#vulkan).

## Installation

For the updated GitHub build artifact (already includes `custom` and the Vulkan early loader), follow [build/install and OptiScaler compatibility notes](docs/OPTISCALER_BUILD_VI.md).

1. Download dlssg-to-xefg-#.zip from the [latest release](https://github.com/realdody/dlssg-to-xefg/releases/latest).
2. Make sure you're using OptiScaler v10 https://github.com/optiscaler/OptiScaler-nightly/releases/latest.
3. Extract the XeMFG XMX folder from the zip into OptiScaler\custom. The folder structure should be: gamefolderwithoptiscaler'sdll\OptiScaler\custom\XeMFG XMX
4. Launch the game and open OptiScaler with insert. 
5. The recommended FG input is DLSSG via NVNGX, as it lets you use XeMFG XMX as an output directly.
   If DLSSG via NVNGX is not available or doesn't work, you can use any other input by setting the FG Output to DLSSG, and the FG replacement to XeMFG XMX. Then press Save Settings in the bottom right to save

7. Make sure "Current DLSSG state" says "ON" with the fake frame count you selected. Then check \OptiScaler\custom\XeMFG XMX\dlssg\dlssg_to_xefg_intel_is_better.log
   It should say "Ported XMX MFG model installed, one optical flow field per generated batch" and "Creating XeFG context: 2560x1440, format: 28, flags: 1, ui mode: 4, model: 16"

Model 16 means it's running the XMX path.

## Which libxess_fg.dll is used

1. The `libxess_fg.dll` next to this dll
2. A `libxess_fg.dll` up to three folders above this dll, which covers OptiScaler's own copy in its root
   (`\OptiScaler\libxess_fg.dll`)
3. Every `libxess_fg.dll` already loaded in the process, the first compatible one wins

The copies from 1 and 2 are used whether or not they are already loaded. They are the patches' intended target:
MFG-Unlocker's port rewrites the model catalog and redirects the program decoder, which has to happen before XeFG's
first context. A copy the game itself loaded has normally passed that point, and patching it leaves the copy
OptiScaler initializes untouched. The usual result is `Failed to create heap` from `Init`. Such a copy is only a
fallback for installs without a `libxess_fg.dll` next to this dll.

Compatible means a build with known function addresses (XeFG 1.3.1 from XeSS SDK 3.0.2, XeFG 1.3.4), or any XeFG
1.3.x where the functions can be found by scanning and the settings match one of the known layouts. The struct layouts
were only verified for 1.3.1 and 1.3.4.

`dlssg_to_xefg_intel_is_better.log` next to the dll says which one was picked.

### Checking a libxess_fg.dll

`python tools/CheckXeFgBuild.py libxess_fg.dll` says, from the file alone (no GPU, no Windows, nothing is run), what a build gives
this dll; add `--vulkan-programs <dll folder>\xefg_mfg_programs\vulkan` to check the Vulkan models against it too. It needs Python
3.8 or newer and a zstd library (`pip install zstandard`, or `libzstd` for ctypes), because XeFG packs its compute programs with
zstd and two dictionaries of its own. It checks:

- the build: version, PE time stamp and size, SHA-256, and that the function addresses of `knownBuilds` (`src/XeFG.cpp`) are the
  ones the scan `XeFG::Scan` makes for an unknown build finds;
- that all of the build's compute programs decode, and which of the 13 Intel programs of the tensor-core route's layer table
  (`src/MfgPort/Cuda/cuda_layers.h`) and of the Intel programs of the Vulkan models it has, byte for byte (SHA-256 of the DXBC
  container, which is how both routes recognize a program);
- that MFG-Unlocker's port knows the build: its instruction family is found (`src/MfgPort/RuntimeLayout.generated.h`), the call
  site `ShaderPort` redirects is a call to XeFG's program decoder (which, like the vector resize it uses, starts as expected), and
  the model catalog `ModelPort` rewrites is in the file's data, with the donor model's name.

Exit code 0 means everything is where the repository expects it, 1 that something isn't (the output says what, and which route
won't work), 2 that the file or the zstd library can't be used. `--selftest` only checks the zstd decoder, on known answers.
Checked this way, XeFG 1.3.4.95 (22,866,056 bytes, SHA-256 `752b1491…`) passes everything: it has all 13 Intel programs of the
tensor-core layer table and all 56 Intel programs of the Vulkan models, byte for byte, and the port's instruction family `fg95`.
That is a static result: running it in a game on a Windows machine is not what this checks.

## Vulkan

Source hardening changes and validation limits: [Vietnamese comparison and test report](docs/VULKAN_HARDENING_VI.md).

XeFG only has a D3D12 API. This dll runs it on a Vulkan game's `VkDevice` by itself: the D3D12-on-Vulkan layer of
[MFG-Unlocker](https://github.com/dashdogy/MFG-Unlocker) (MIT, its V1.7.1 source) is built into it (`src/vulkan/`), together
with the hooks that make the game's Vulkan instance and device fit for it. XeFG's compute programs run on that layer as SPIR-V.
Those are the Vulkan models, 83 programs, the same in all four 1.7.1 builds of MFG-Unlocker (1.7.1-beta2fix.1, 1.7.1-beta.3,
1.7.1-rc.2 and the final 1.7.1.177), see [docs/VULKAN-MODELS.md](docs/VULKAN-MODELS.md). The dll reads them from
`xefg_mfg_programs\vulkan` next to itself. `MFGUnlocker.asi` isn't needed to run any of this. On an NVIDIA RTX GPU the ported
model's int8 layers can run on the tensor cores instead, see [Tensor cores (NVIDIA)](#tensor-cores-nvidia).

1. Install the dll as above, in `custom\XeMFG XMX\dlssg`, and put the Vulkan models in the `xefg_mfg_programs` folder next to it,
   as `xefg_mfg_programs\vulkan`, see [The Vulkan models](#the-vulkan-models). Put Intel's `libxess_fg.dll` where it's looked for,
   as for DX12.
2. Start the Vulkan game and turn frame generation on. The **Vulkan** block of the plugin's tab (PLUGINS in OptiScaler's menu)
   says *Ready*, or *Active* while frames are generated, with the largest multiplier and the model (`Up to 6x · port model`).
   The log, `dlssg_to_xefg_intel_is_better.log`, has the same in lines that start with `Vulkan`.
3. If it says *Unavailable* with "the game created its VkDevice before this dll's hooks were installed" or "the game's VkInstance
   wasn't seen", the game created its Vulkan device before the dll was loaded: see [Early start](#early-start).

The GPU and driver have to offer what XeFG's programs use: Vulkan 1.3 with timeline semaphores, synchronization2, maintenance4 and
shader integer dot product, extended storage image formats, subgroup arithmetic in compute shaders and `VK_KHR_push_descriptor`.
Intel GPUs are turned down (XeFG's own network needs Intel's D3D12 driver path there). When a game can't have frame generation the
tab and the log say why, in words; the reasons are listed in `src/vulkan/VulkanPolicy.h`. DX12 games aren't affected either way.

### Early start

The hooks add what the programs need to the game's `VkInstance` and `VkDevice` *when the game creates them*, so they have to be
installed before that. OptiScaler loads a custom frame generation dll when the game asks for DLSS Frame Generation, which for a
Vulkan game can be after the device exists. When that happens the log says so and the tab says *Unavailable*. For those games
start the hooks earlier, with a copy of this dll as an OptiScaler plugin:

1. Copy `dlssg_to_xefg_intel_is_better.dll` to OptiScaler's `plugins` folder and rename the copy to
   `dlssg_to_xefg_intel_is_better.asi`. (It is this dll under another extension, not MFG-Unlocker's file.)
2. Set `LoadAsiPlugins = true` under `[Plugins]` in `OptiScaler.ini`.

OptiScaler then loads the copy when it starts and calls its `InitializeASI`. The copy loads the real dll from
`custom\<name>\dlssg\dlssg_to_xefg_intel_is_better.dll` and starts its hooks; it has no tab, settings or models of its own. Keep
both copies the same build. A game that creates its device after OptiScaler loaded the dll needs none of this.

`XEFG_VULKAN_HOOKS=0` in the environment, `VulkanHooks = false` in `[Plugin.<name>]` of `OptiScaler.ini` (the dll's own section, see
[OptiScaler menu](#optiscaler-menu)) or the tab's **Vulkan hooks** checkbox leave the game's Vulkan device alone. The hooks only change what a game's Vulkan instance and device
are created with (API version, the features and the extension above), nothing else of the game's.

### The Vulkan models

`xefg_mfg_programs\vulkan` has to be `custom\XeMFG XMX\dlssg\xefg_mfg_programs\vulkan\programs.tsv` with the 83 `<id>.spv`
beside it (and the three `input_copy_<n>.spv`, a copy of the shaders built into the dll, which it doesn't need). The dll reads the
folder when OptiScaler loads it and checks every program against the SHA-256 in `programs.tsv`. The log then says
`Vulkan models: 83 programs (...), ... MiB, from MFG Unlocker 1.7.1 (file version 1.7.1.177) in ...`, and the **Vulkan** block of the
tab says `Vulkan models: 83 programs`, or `Vulkan models: not installed` without the folder, or in red
`Vulkan models: can't be used` with the reasons in the tooltip.

`tools/ExtractXeFgVulkanPrograms.py MFGUnlocker.asi --output <dll folder>\xefg_mfg_programs\vulkan` makes the folder from an
`MFGUnlocker.asi` 1.7.1 or newer (Python 3.8 or newer, standard library only), once; the asi isn't needed after that.
`tools/InspectMfgUnlockerAsi.py MFGUnlocker.asi` says whether an asi has what this dll relies on. The models are Intel derived
program data, so they aren't in the repository, see [NOTICE.md](NOTICE.md).

### Choosing the model

The tab's network path picks the model on Vulkan too, and XeFG's context is created again when it changes:

| Network path | Vulkan model | Its int8 layers | The tab shows |
|---|---|---|---|
| XMX Port | the ported MFG model | SPIR-V programs | `port` |
| XMX (native), CUDA | the ported MFG model | the tensor cores of an NVIDIA RTX GPU when the route can run, else SPIR-V programs | `port`, and the **Tensor cores** line |
| Fallback (DP4a) | XeFG's own DP4a model | SPIR-V programs | `DP4a` |

It is the ported model whenever the DXIL program set next to the dll has it. XeFG's own XMX network isn't used on Vulkan, so XMX
(native) means the same as CUDA there, as it does in the D3D12 route on a GPU without XMX. The default network path is XMX (native).

### Tensor cores (NVIDIA)

On an NVIDIA RTX GPU the ported model's int8 layers run as CUDA kernels on the tensor cores, as they do in the D3D12 route
(`src/MfgPort/Cuda/`). The 16 kernels are the same, from `xefg_mfg_programs\xefg_cuda_layers.fatbin` (the pinned build, SHA-256
`2ffae52e…`, the file the D3D12 build already puts there). On Vulkan they are launched with `VK_NVX_binary_import` in the command
buffer the game gave NGX, and the images they read and write are named with `VK_NVX_image_view_handle`. The launching is
MFG-Unlocker's `xvk_tensor.cpp` (V1.7.1), vendored with the rest of its layer; `src/vulkan/VulkanTensor.cpp` is what
MFG-Unlocker's host does around it, for this dll.

What it takes:

- An NVIDIA GPU of compute capability 7.5 (RTX 20 series), 8.6, 8.9 or 12.0 (RTX 30, 40 and 50 series), and a driver that offers
  `VK_NVX_binary_import` (version 2 or newer) and `VK_NVX_image_view_handle` (version 3 or newer).
- `xefg_cuda_layers.fatbin` in `xefg_mfg_programs`, with all 16 kernels.
- The network path XMX (native) or CUDA, the first being the default. The XMX Port path keeps the layers on their SPIR-V programs.

A device has to have the two extensions when it is created, so the hooks add them to an NVIDIA GPU's `VkDevice` as they add push
descriptors, unless `XEFG_TENSOR_CORES` is off. If the driver refuses the device with them, the hooks create it again without them and
the SPIR-V programs run everything. A layer the route doesn't take (a shape it doesn't know, an image view it can't name) keeps its
SPIR-V program in the same frame, so a frame never lacks a layer. If the driver crashes in one of the route's calls, the route turns
itself off for that device and the SPIR-V programs run the layers.

| Environment | |
|---|---|
| `XEFG_TENSOR_CORES=0` (or `off`, `false`) | Turns the route off, in D3D12 as in Vulkan; the hooks then don't add the NVX extensions either |
| `XEFG_TENSOR_CORES_MASK=<hex>` | The layers the Vulkan route may launch, a bit per layer of the table of 17 (the lowest for the first), default `1FFFF`. To find a layer that misbehaves |
| `XEFG_TENSOR_CORES_VULKAN_BARRIERS=1` | A full barrier after each launch, for diagnosis; slower |

The **Vulkan** block of the tab has a **Tensor cores** line: `on · sm_89` (the compute capability) once layers have run there, `idle`
while none has (no frame generated yet, or none of this XeFG build's programs is in the route's layer table, which as in the D3D12
route is that of `libxess_fg.dll` 1.3.1, XeSS SDK 3.0.2, and 1.3.4.95 has the same 13 Intel programs byte for byte; another build
may not, `tools/CheckXeFgBuild.py` says), `ready` when the route is installed but the XMX Port or Fallback path is
selected, `off`, `unavailable` (the tooltip says why, see
`VulkanPolicy::DescribeTensor` in `src/vulkan/VulkanPolicy.h`), `NVIDIA RTX GPUs only`, or in red `the driver faulted`. Its tooltip counts
the layers launched on the tensor cores and the ones left to SPIR-V. In the log, lines that start with `Vulkan tensor cores` and
the `cuda_route_vk` and `cuda_vk_fallback` ones (the reason a layer stayed on SPIR-V: `dispatch_shape`, `bindings_incomplete`,
`view_type`, `image_handle`, `shadow_…`, `launch` or `exception`).

The route is on by default on an RTX GPU, for parity with the D3D12 route and with MFG-Unlocker. It has not run on an NVIDIA GPU yet,
see [Status](#status): if a game misbehaves with it, `XEFG_TENSOR_CORES=0`, or XMX Port in the tab, brings back the SPIR-V programs.

### Not in this port

What MFG-Unlocker's own Vulkan provider has and this dll leaves out: its GPU timing and its registration as an OptiScaler provider
(this dll already is OptiScaler's frame generation dll). The asi's own settings (`XeFGModel`, `XeFGTensorCores`, `XeFGTensorCoresMask`
and `XeFGTensorCoresVulkanBarriers` in `[XeFG]` of `OptiScaler.ini`) aren't read: the tab's network path chooses the model and whether
the tensor cores are used, and the environment variables above are the other switches.

A build without the Vulkan headers has no runtime of its own and goes through `MFGUnlocker.asi` as it did before (the tab says
`MFGUnlocker.asi isn't loaded` or shows what the asi reports); see [Build](#build).

### Status

Not built with MSVC or run in a game yet: where this was written there was no Windows, no Windows SDK and no GPU. What has been
checked: the models reader and the program sets against all four builds' extracted folders (`tests/vulkan_programs`), the Vulkan
exports and how they hand a device to the runtime (`tests/vulkan_provider`), the runtime's decisions and reason texts, the layer
tables of the two tensor-core routes against each other and the fatbin's SHA-256 and kernels (`check_layer_tables.py`), and a syntax
check of the Windows-only sources against stand-ins for the Win32, D3D12 and Vulkan APIs (`tests/vulkan_runtime`). The first
build on GitHub Actions is the first compile against the real headers: if it fails, the log of its **Build** step has the compiler's
messages. Whether OptiScaler loads the dll in time without the early copy depends on the game and OptiScaler's version, and how the
hooks get along with other hooks on `vkCreateInstance` and `vkCreateDevice` (an overlay's, MFGUnlocker.asi's if it is loaded too)
hasn't been tried. Neither has the tensor-core route on Vulkan: no NVIDIA GPU, so no kernel has been launched through
`VK_NVX_binary_import` with this dll, and whether NVIDIA's driver takes the game's own command buffers with it is for a game to
show. That route is MFG-Unlocker's code as it is (the changes are the include paths, `tools/VendorXvk.py --check` lists them).

## OptiScaler menu

Loaded by OptiScaler from `custom\<name>\dlssg`, the dll adds a **dlssg-to-xefg** tab under PLUGINS with XeFG's status
and its settings, kept in `[Plugin.<name>]` of `OptiScaler.ini`. Its **Vulkan** block shows the runtime's state (idle, ready, active
or unavailable and why), the hooks, and the Vulkan models found next to the dll.

## Build

Visual Studio 2026 (v145 toolset), open `dlssg-to-xefg.sln` and build Release x64. The output lands in `bin\Release`.

The Vulkan runtime needs the Khronos Vulkan headers (MFG-Unlocker builds against 1.4.363): pass
`-p:VulkanIncludeDir=<folder that has vulkan\vulkan.h>`, the `include` folder of a
[Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) checkout, or have the Vulkan SDK installed (`VULKAN_SDK`).
Without them the build warns `Vulkan runtime: NOT built` and makes a dll that has none, which sends Vulkan games to
`MFGUnlocker.asi`; the build says `Vulkan runtime: built in` when it has it. The layer's sources and MinHook are in the repository
(`src/vulkan`, `third_party/minhook`, see [NOTICE.md](NOTICE.md)).

GitHub Actions builds it too. `build.yml` runs on every push and pull request: it clones the Vulkan headers, builds Release x64
(and fails if the dll came out without the runtime), checks that the dll exports every function the sources declare
(`tools/CheckExports.py`, the OptiScaler plugin entries `OptiPluginInit` and `InitializeASI` among them) and keeps
`XeMFG XMX\dlssg\...` as a workflow artifact, with the licenses a binary has to carry. Three Linux jobs run the tests of the
Vulkan provider, the Vulkan runtime's decisions and syntax (and those of `tools/CheckXeFgBuild.py`, `tests/xefg_build`), and the
models reader. `release.yml` still publishes a release by hand, built the same way.

The ported XeSS MFG programs are not committed, and no MFG-Unlocker checkout or submodule is needed: the build
downloads its pinned v1.7.0 release dll, extracts the program set and the validated tensor-core fatbin into
`bin\Release\xefg_mfg_programs`, which has to ship next to the dll. The download and its contents are checked against
pinned SHA-256 values and cached under `obj\Release\packages`. The generated folder also carries MFG-Unlocker's MIT
LICENSE. For an offline build, pass `/p:MfgUnlockerDll=<path to MFGUnlocker.dll>`. No local CUDA toolkit is needed:
`xefg_cuda_layers.fatbin` is the pinned RCDATA 7300 resource (`2ffae52e…`) from that dll.

The Vulkan models aren't committed either, and neither is the asi they come out of. Where the build takes them from, first
that exists:

- locally, `/p:MfgUnlockerAsi=<path to MFGUnlocker.asi>` writes them to `bin\Release\xefg_mfg_programs\vulkan` (Python 3.8 or
  newer on PATH, `/p:PythonExe=` picks another one);
- in GitHub Actions, the folder `third_party/xefg_mfg_programs/vulkan` (what `tools/ExtractXeFgVulkanPrograms.py` writes, from the
  current extractor), or an `MFGUnlocker.asi` at `third_party/MFGUnlocker.asi` that `build.yml` checks with
  `tools/InspectMfgUnlockerAsi.py` and extracts. Either way the artifact has the models next to the dll and the models test runs on
  them. They are Intel derived program data, so commit them to a private repository only.

Without either the dll is built the same and its tab says the Vulkan models aren't installed: copy the folder next to the dll.

A `libxess_fg.dll` at `third_party/libxess_fg.dll` is checked too, by the models test: it runs `tools/CheckXeFgBuild.py` on it, with
the models of the steps above when there are any (see [Checking a libxess_fg.dll](#checking-a-libxess_fgdll)). It is Intel's, so the
same applies: `*.dll` is in `.gitignore` on purpose, and only a private repository should get it (`git add -f`).

The tensor-core route is optional at runtime. The fatbin has SASS for RTX 20/30/40/50 and is loaded next to the program
set; no CUDA runtime is linked or shipped. On an RTX GPU the log shows `cuda_route active`, then `cuda_layer` lines,
with `cuda_dispatch_fallback` for any layer it had to leave on DXIL. The layer hashes are those of XeFG 3.0.2's programs
(`libxess_fg.dll` 1.3.1), and XeFG 1.3.4.95 carries the same 13 Intel programs byte for byte, so both builds get the same layers
(a build that doesn't have them keeps those layers on DXIL; see [Checking a libxess_fg.dll](#checking-a-libxess_fgdll)). Set
`XEFG_TENSOR_CORES=0` to force the DXIL path without
deleting the fatbin. `CudaRoot` is only for rebuilding the kernels from source with CUDA 13.2 during development; it
overwrites the extracted fatbin in the build output. The same fatbin serves the Vulkan route, see
[Tensor cores (NVIDIA)](#tensor-cores-nvidia); `tests/vulkan_runtime/check_layer_tables.py --fatbin <file>` checks it (the pinned
SHA-256, all 16 kernels) and that the two routes' layer tables agree, `build.yml` runs it on the build's output.

### Trying the tensor-core path

1. Build Release normally; the pinned fatbin is extracted automatically. Only set `CudaRoot` if you changed the kernels.
2. Keep `libxess_fg.dll` 1.3.1 (XeSS SDK 3.0.2) or 1.3.4.95 next to the dll: the route's layer hashes are of the programs that
   both carry. For another build, `python tools/CheckXeFgBuild.py libxess_fg.dll` says whether it has them.
3. Launch, then check the log for `cuda_route active`, `"compute_capability":"12.0"` on an RTX 50 card, and
   `cuda_layer` lines. `cuda_dispatch_fallback` means that layer stayed on DXIL by design or because its bindings
   were incomplete.
4. If the log says `Tensor-core route inactive: <reason>`, send that line and the surrounding `cuda_route` lines; the
   reason distinguishes a missing fatbin, a non-NVIDIA/unsupported adapter, a wrapped device, and NVAPI failure.
5. For a Vulkan game, look for `Vulkan tensor cores: ready on compute capability 12.0, 17 layers` and the `cuda_route_vk` lines
   (`"state":"active"`); the **Tensor cores** line of the tab says `on · sm_120`, and its tooltip counts the layers launched. When it says
   unavailable, the line `Vulkan tensor cores: unavailable, <reason>` is the thing to send, with the `Vulkan device … prepared for XeFG`
   line (it says whether the device got the NVX extensions), the `Vulkan XeMFG ready` line and any `cuda_vk_fallback` lines.

## Credits

The port in `src/MfgPort/` and `src/MfgKernels.cpp` is from [MFG-Unlocker](https://github.com/dashdogy/MFG-Unlocker),
MIT licensed, Copyright (c) 2026 Michael Robles. The CUDA route and kernels in `src/MfgPort/Cuda/` are also from
MFG-Unlocker. The Vulkan models and the D3D12-on-Vulkan layer that runs them (`src/vulkan/`), with its tensor-core launcher
(`xvk_tensor.cpp`), are MFG-Unlocker's too, as are the Vulkan provider this dll's runtime is ported from; the hooks use [MinHook](https://github.com/TsudaKageyu/minhook). See
[NOTICE.md](NOTICE.md).

## License

GPLv3, see [LICENSE](LICENSE).
