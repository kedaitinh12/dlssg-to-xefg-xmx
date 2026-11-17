# Vulkan models

The Vulkan models are the SPIR-V programs the Vulkan runtime built into this dll runs XeFG with on a game's `VkDevice`. The dll
reads them from `xefg_mfg_programs\vulkan` next to itself, checks them and shows them in its tab. `MFGUnlocker.asi` isn't needed
to run them, it is only where the folder is made from, once: see [Making the folder](#making-the-folder). Read off four 1.7.1
builds of `MFGUnlocker.asi`, see [Builds](#builds).

## What the models are

The asi embeds 83 compute programs as SPIR-V 1.6, 5,356,728 bytes in all, LZMA-compressed in its resource 3904 (ids 7400 to
7482). Each one is the translation of a DXIL program that XeFG hands to `CreateComputePipelineState`. MFG-Unlocker's
D3D12-on-Vulkan layer (xvk) looks the translation up by the SHA-256 of that DXIL container, and a miss fails closed. So the
set is the Vulkan twin of the DXIL programs the D3D12 route uses. It is a table the layer reads, not a model file a caller can
load.

| Origin | Count | What |
|---|---:|---|
| port | 25 | Translations of DXIL programs of the XMX Port set, the ported MFG model: 25 of the set's 30 distinct DXIL containers, which stand for 45 of its 54 program keys |
| port-extra | 1 | id 7433, the translation of one of the 7 replacement shaders MFG-Unlocker embeds beside the port set (resource 3902). The layer maps the other six to the Intel program each replaces (`kReplacements` in `src/vulkan/xvk_programs.cpp`) |
| intel | 56 | Intel's XeFG programs, translated as XeFG creates them |
| unhashed | 1 | id 7482, 162,224 bytes, the asi lists no DXIL source for it |

MFG-Unlocker's generator (`generate_xvk_programs.py`) notes that 4 of the 83 were added for inverted depth and 18 for XeFG's AMD
selection and the 2x model route.

What the modules need from the device, read from them: every one uses `Shader`; 57 use `DotProduct` and
`DotProductInput4x8BitPacked` with `SPV_KHR_integer_dot_product`; 20 use `StorageImageExtendedFormats`; one uses
`GroupNonUniform` and `GroupNonUniformArithmetic`. None uses cooperative matrices. Entry point `main`, 484 bindings in
all (contiguous per program), push constants of 16 to 204 bytes.

## Where the dll reads them

From `xefg_mfg_programs\vulkan`, a folder of its own inside the DXIL set's `xefg_mfg_programs` (the manifest of the DXIL set
is that folder's `programs.tsv`, so the two can't share one):

```text
custom\XeMFG XMX\dlssg\
    dlssg_to_xefg_intel_is_better.dll
    xefg_mfg_programs\
        programs.tsv, <key>.dxil, xefg_cuda_layers.fatbin, LICENSE     the DXIL set, D3D12
        vulkan\
            programs.tsv                 the manifest, one row per program (columns below)
            7400.spv ... 7482.spv        the SPIR-V modules
            input_copy_0.spv ... _2.spv  the three input-copy shaders: a copy of the ones built into the dll, which doesn't read these
            SOURCE.txt                   the MFGUnlocker.asi it was made from
            LICENSE                      MFG-Unlocker's MIT license
            XEFG-PROGRAMS-NOTICE.txt     MFG-Unlocker's note on the Intel derived program data
```

When OptiScaler loads the plugin (`OptiPluginInit`) the dll reads the manifest and checks every program: the file is there,
has the size the manifest gives, is a SPIR-V module and hashes to the SHA-256 the manifest lists. Nothing in the folder is
executed and a program that fails any of it makes the whole set unusable, so a half copied folder can't pass for a good one. The
layer reads each module again when XeFG creates its pipeline and checks the hash a second time. The answer goes to
`dlssg_to_xefg_intel_is_better.log`:

```text
Vulkan models: 83 programs (port 25, port-extra 1, intel 56, unhashed 1), 5.1 MiB, from MFG Unlocker 1.7.1 (file version 1.7.1.177) in ...\xefg_mfg_programs\vulkan
```

and to the **Vulkan** block of the plugin's tab: `Vulkan models: 83 programs`, grey `Vulkan models: not installed` when
there is no folder (DX12 games don't need it, nor does anything else break), or red `Vulkan models: can't be used` with the
reasons in the tooltip (a missing file, a wrong size, a hash that doesn't match, a manifest row it can't read).

`programs.tsv` columns: id, SPIR-V SHA-256, bytes, push constant bytes, bindings, local size (`-` when unknown), SHA-256 of the
DXIL program it translates (`-` when none), origin, keys of the XMX Port programs it translates (`-` when none), and the
descriptors as `binding:type` pairs in binding order, which the layer builds its descriptor set layouts from (type 0 storage
buffer, 1 uniform buffer, 2 sampled image, 3 storage image, 4 sampler; D3D12 SRVs are numbered from 0, UAVs from 64, CBVs from
128 and samplers from 192). The reader is `src/VulkanPrograms.cpp`, plain C++ with the SHA-256 passed in.

## Builds

| `MFGUnlocker.asi` | File version | SHA-256 |
|---|---|---|
| 1.7.1-beta2fix.1 | 1.7.1.156 | `8748f449…e88e5` |
| 1.7.1-beta.3 | 1.7.1.165 | `095a0321…5505f` |
| 1.7.1-rc.2 | 1.7.1.173 | `171ac18d…6a4f8` |
| 1.7.1 | 1.7.1.177 | `5a389e6c…0aa74` |

All four carry the same 83 programs and the same three input-copy shaders: every `.spv` and `programs.tsv` are byte for byte
the same (`programs.tsv` is `18ce39ef3e69c064ba46ee70738b2e9cbf13e5fb000780e7061efc0543724079`).
So is the XMX Port DXIL set they embed: its manifest hashes to `d74bd783d29280ed372653701ce9b0395362e381faace1213206e0bdd9bfc9e5`, which is the one
`tools/GenerateXeFgPrograms.ps1` pins for `xefg_mfg_programs`. The DXIL programs the Vulkan models are keyed by are therefore the
ones this dll's D3D12 route installs. Their exports and their provider table are the same too (size 0x68, version 1, the same
slots, only the addresses move), which `tools/InspectMfgUnlockerAsi.py` checks. It reads the files and runs nothing.

What the builds differ in, from the strings in their binaries and not from release notes: 1.7.1-beta.3 adds a CUDA tensor-core
route on Vulkan (`VK_NVX_binary_import`, the settings `XeFGTensorCoresMask` and `XeFGTensorCoresVulkanBarriers`) and the
`DLSSG.DepthSubrect` and `DLSSG.MVecsSubrect` inputs; 1.7.1-rc.2 registers the asi itself as OptiScaler's FG Nvngx provider, "no
shim file"; the final 1.7.1.177 has all of that. This dll's runtime has the tensor-core route on Vulkan, from the 1.7.1 source
(see [Tensor cores (NVIDIA)](../README.md#tensor-cores-nvidia)); what the newer inputs would change in it hasn't been looked at.

## What runs them

The network path in the tab (XMX Port, CUDA, Fallback, XMX) says how this dll's D3D12 route runs XeFG, by patching XeFG's own
D3D12 context. A Vulkan game has no D3D12 device for that, so the dll does what MFG-Unlocker does: it gives `libxess_fg.dll` a
D3D12 API implemented on Vulkan, and these SPIR-V programs are what that implementation dispatches in place of the DXIL. All of
it is in the dll (`src/vulkan/`):

1. the D3D12-on-Vulkan layer (`xvk*.cpp`), MFG-Unlocker's own code from its V1.7.1 source of 13 November 2026. It creates the
   pipelines, descriptors and resources `libxess_fg.dll` asks for and looks each program up by the SHA-256 of the DXIL it was
   given. Three of its files are patched to read the programs from `xefg_mfg_programs\vulkan` and not from the resources of a
   dll, and three include the tensor-core headers from where they are in this repository; `src/vulkan/VENDORED.txt` lists every
   file with the SHA-256 of the original and what was done to it;
2. the hooks (`VulkanHost.cpp`, MinHook on the Vulkan loader's `vkCreateInstance` and `vkCreateDevice`): when the game creates its
   instance and its device they raise the instance's API version to 1.3 where the loader allows it and switch on what the
   programs need, `VK_KHR_push_descriptor` and the features timeline semaphore, synchronization2, maintenance4, shader integer
   dot product and extended storage image formats, and on an NVIDIA GPU `VK_NVX_binary_import` and `VK_NVX_image_view_handle`
   for the tensor-core route. Nothing the game asked for is taken away. They have to be in before the game creates its device,
   see [Early start](../README.md#early-start);
3. the runtime (`VulkanRuntime.cpp`), MFG-Unlocker's Vulkan provider ported to this repository: it answers the NGX calls, creates
   XeFG's context on the layer and records the D3D12 command lists XeFG executes into the game's own `VkCommandBuffer`;
4. the tensor-core route (`xvk_tensor.cpp`, MFG-Unlocker's, and `VulkanTensor.cpp`): on an NVIDIA RTX GPU the int8 layers of the
   ported model are launched as the CUDA kernels of `xefg_mfg_programs\xefg_cuda_layers.fatbin` in place of their SPIR-V
   programs, layer by layer, the SPIR-V program staying where the route can't take a layer;
5. the `NVSDK_NGX_VULKAN_*` exports (`ExportsVulkan.cpp`, `VulkanProvider.cpp`) that OptiScaler or the game calls, the same
   twelve as `nvngx_dlssg.dll`.

What the network path does on Vulkan:

| Network path | Model on Vulkan | Int8 layers |
|---|---|---|
| XMX Port | the ported MFG model (model 16), which runs the `port` programs above, as long as the DXIL set is installed next to the dll | the SPIR-V programs |
| XMX (native), CUDA | the same | the tensor cores of an NVIDIA RTX GPU when the route can run, else the SPIR-V programs |
| Fallback (DP4a) | XeFG's own DP4a model (model 18) | the SPIR-V programs |

No module in the set uses cooperative matrices, so Intel's XMX network has no counterpart in it: XMX (native) runs the ported model,
on the tensor cores where there are any the route can use. MFG-Unlocker's provider also has GPU timing and registers itself as
OptiScaler's FG provider, which this dll leaves out (it already is OptiScaler's frame generation dll). DirectML isn't involved.

## Status

Not compiled with MSVC and not run in a game: where this was written there was no Windows, no Windows SDK and no GPU. What has
been checked, on Linux and in CI:

- `tests/vulkan_programs`: the reader of the models against synthetic sets, broken ones included, and against the sets extracted
  from the four builds;
- `tests/vulkan_provider`: the Vulkan exports and how they hand a device to the runtime (and, for a build of the dll without the
  runtime, to the provider table of an asi), against fakes;
- `tests/vulkan_runtime`: the runtime's decisions (`VulkanPolicy.h`, the tensor-core switches and layer mask among them), that every
  reason it can give has a text, that the tensor-core layer tables of the D3D12 and the Vulkan route say the same and that the
  fatbin is the pinned one with all 16 kernels, and a syntax check of its Windows-only sources against stand-ins for the Win32, D3D12
  and Vulkan headers with clang++ and g++.

They cover this dll's side and nothing a GPU does with it. The first build on GitHub Actions is the first compile against the
real headers. Not tried: how the hooks get along with other hooks on `vkCreateInstance` and `vkCreateDevice` (an overlay's, a
loaded `MFGUnlocker.asi`'s), whether OptiScaler loads the dll early enough without the copy described under
[Early start](../README.md#early-start), and the tensor-core route on a GPU: no NVIDIA GPU was at hand, so no kernel has been
launched through `VK_NVX_binary_import` with this dll.

## Making the folder

```
python tools/ExtractXeFgVulkanPrograms.py MFGUnlocker.asi --output "custom\XeMFG XMX\dlssg\xefg_mfg_programs\vulkan"
python tools/InspectMfgUnlockerAsi.py MFGUnlocker.asi
```

Python 3.8 or newer, standard library only. The extractor writes the files listed above after checking every module against
the SHA-256 the asi lists for it, against its own SPIR-V structure, against the DXIL set the asi embeds and the descriptor types
against the asi's own binding table, and the input-copy shaders against the SHA-256 their generator pins. The tables are found by
scanning, so a build it was never run on works when its layout is the same, and `--only-known` refuses a build that isn't one of
the four above. The inspector reads the asi the same way and says whether the Vulkan models can be made from it and, for a build
of the dll without the Vulkan runtime, whether the provider table the Vulkan exports forward to has the layout they expect. Its
exit code is 0 when the models can be made.

The build does the same when it is given an asi: `msbuild dlssg-to-xefg.sln -p:Configuration=Release -p:Platform=x64
-p:MfgUnlockerAsi=C:\path\MFGUnlocker.asi` writes `bin\Release\xefg_mfg_programs\vulkan`. `build.yml` uses the folder
`third_party/xefg_mfg_programs/vulkan` when the repository has one, else `third_party/MFGUnlocker.asi`. The output is program
data derived from Intel's XeSS SDK, so like the DXIL set it stays out of the repository (see [NOTICE.md](../NOTICE.md)).

## The set

Hashes are cut to 12 characters, `programs.tsv` has them whole. `push` is the push constant size in bytes.

| id | origin | bytes | push | bindings | local size | SPIR-V SHA-256 | DXIL SHA-256 |
|---|---|---:|---:|---:|---|---|---|
| 7400 | port | 4,088 | 16 | 3 | 8x8x1 | `b0d451992246…` | `0142035fb084…` |
| 7401 | intel | 56,344 | 48 | 6 | 8x8x1 | `ae78cb701e84…` | `01c8beac27d3…` |
| 7402 | intel | 117,752 | 16 | 6 | 16x16x1 | `5236409256d3…` | `09887269759e…` |
| 7403 | intel | 74,064 | 48 | 5 | 8x8x1 | `6469385cf871…` | `0be5240659a6…` |
| 7404 | port | 4,180 | 24 | 2 | 8x8x1 | `4b000533f3de…` | `13abff5d769e…` |
| 7405 | port | 74,064 | 48 | 5 | 8x8x1 | `ab0be8654a84…` | `1bebcab19371…` |
| 7406 | port | 74,904 | 48 | 5 | 8x8x1 | `fa9b4f5b4eb5…` | `1d07fc625b5a…` |
| 7407 | intel | 6,944 | 52 | 3 | 8x8x1 | `965c05ccb517…` | `2be81de8fa38…` |
| 7408 | port | 65,540 | 48 | 5 | 8x8x1 | `44a701aa1833…` | `2f31a5964c89…` |
| 7409 | port | 51,528 | 48 | 6 | 8x8x1 | `c6d5e4f34cb0…` | `2f338a641e65…` |
| 7410 | intel | 42,984 | 48 | 5 | 8x8x1 | `483418dc9152…` | `338f19578972…` |
| 7411 | port | 74,544 | 48 | 5 | 8x8x1 | `ec42aa61047a…` | `3577b7537200…` |
| 7412 | port | 36,012 | 48 | 5 | 8x8x1 | `134f9d4f52fb…` | `397171dd727f…` |
| 7413 | intel | 38,168 | 48 | 5 | 8x8x1 | `ed696f453236…` | `3cdf1a4746fc…` |
| 7414 | intel | 102,472 | 16 | 5 | 4x8x1 | `8804de2804ed…` | `3eb241c8b60e…` |
| 7415 | intel | 9,860 | 40 | 4 | 8x8x1 | `5d5dca9f0d7e…` | `3f3f91d2ecd4…` |
| 7416 | intel | 8,496 | 40 | 3 | 8x16x1 | `2f93b611dd46…` | `4008095b006a…` |
| 7417 | port | 195,924 | 204 | 16 | 8x16x1 | `1a13ab5cebdc…` | `402afbef69b7…` |
| 7418 | intel | 9,612 | 36 | 4 | 8x8x1 | `ab2b89e14a42…` | `40e159414d5b…` |
| 7419 | intel | 154,020 | 16 | 5 | 8x4x8 | `5b84a0fa7ad0…` | `41fdd0aa080e…` |
| 7420 | intel | 195,604 | 204 | 15 | 8x16x1 | `6168d6cb8524…` | `43c0c713f937…` |
| 7421 | port | 179,180 | 204 | 14 | 8x16x1 | `ed480fb4974c…` | `45c264e344a6…` |
| 7422 | intel | 7,676 | 36 | 4 | 8x8x1 | `d7ec00d1c2c4…` | `4d0dee8aaa33…` |
| 7423 | intel | 195,924 | 204 | 16 | 8x16x1 | `955a904af7ed…` | `4e39a6985f98…` |
| 7424 | port | 11,468 | 52 | 2 | 8x8x1 | `8a178c5a7736…` | `51667b6d9955…` |
| 7425 | intel | 40,664 | 48 | 5 | 8x8x1 | `1f4018a9dd52…` | `51f5754a8f4d…` |
| 7426 | intel | 162,220 | 16 | 5 | 8x4x2 | `0f8879294e1b…` | `589f5a5fc324…` |
| 7427 | intel | 18,300 | 36 | 5 | 8x8x1 | `17598f2eedc3…` | `61e52c4bfa07…` |
| 7428 | intel | 20,064 | 48 | 5 | 8x8x1 | `21c71e26ba30…` | `622cf59690f1…` |
| 7429 | intel | 18,168 | 48 | 5 | 8x8x1 | `7e0669e54b13…` | `64e48214fca0…` |
| 7430 | intel | 179,500 | 204 | 15 | 8x16x1 | `3f18167a12ff…` | `654e0607beb0…` |
| 7431 | intel | 42,984 | 48 | 5 | 8x8x1 | `5580dc11dda3…` | `66904dae2011…` |
| 7432 | port | 65,840 | 48 | 5 | 8x8x1 | `f03a4c6992e1…` | `6acf82efd2a0…` |
| 7433 | port-extra | 65,540 | 48 | 5 | 8x8x1 | `44a701aa1833…` | `6c9bf4b59956…` |
| 7434 | port | 13,516 | 40 | 3 | 8x16x1 | `7e3d96a8127e…` | `6ea243efa335…` |
| 7435 | intel | 8,056 | 56 | 2 | 8x8x1 | `f3424304f39c…` | `79a113c97b8d…` |
| 7436 | intel | 11,108 | 40 | 4 | 8x8x1 | `b4727d84db41…` | `8291bbc36b11…` |
| 7437 | intel | 182,528 | 16 | 6 | 4x8x4 | `f63ff9eb303b…` | `8437587e1759…` |
| 7438 | port | 10,096 | 56 | 2 | 8x8x1 | `2e7a6cee5dac…` | `855ea04032a0…` |
| 7439 | port | 19,344 | 48 | 5 | 8x8x1 | `06374483a71b…` | `8bde7db0638c…` |
| 7440 | intel | 212,452 | 204 | 16 | 8x16x1 | `51c4ce46c1e8…` | `8c1d7667722c…` |
| 7441 | intel | 40,664 | 48 | 5 | 8x8x1 | `7158aae486ae…` | `8c2efb9e955f…` |
| 7442 | port | 133,796 | 16 | 5 | 16x16x1 | `c0fc4a4fe890…` | `8c4a23058529…` |
| 7443 | intel | 36,012 | 48 | 5 | 8x8x1 | `040218bf9bf9…` | `8c72d23e11db…` |
| 7444 | port | 195,604 | 204 | 15 | 8x16x1 | `c1bb4c716331…` | `8cb7e938320f…` |
| 7445 | intel | 212,772 | 204 | 17 | 8x16x1 | `01498d1b7300…` | `8cffea0f5177…` |
| 7446 | intel | 83,508 | 16 | 5 | 4x8x1 | `3f9d2d39b2ef…` | `8e1760008261…` |
| 7447 | intel | 40,664 | 48 | 5 | 8x8x1 | `32887bd5c98e…` | `93506b10b8d3…` |
| 7448 | intel | 2,148 | 24 | 2 | 8x8x1 | `e08342f857b7…` | `9404c9736e1b…` |
| 7449 | intel | 38,168 | 48 | 5 | 8x8x1 | `40fbd5a62018…` | `98375c1ab4e5…` |
| 7450 | intel | 10,096 | 56 | 2 | 8x8x1 | `92f1455853c1…` | `9ef8c15d23bf…` |
| 7451 | intel | 7,848 | 24 | 3 | 8x8x1 | `279bcdfdcd89…` | `a60f7488872a…` |
| 7452 | intel | 182,528 | 16 | 6 | 16x4x2 | `939e94738999…` | `a6aefb7a73e4…` |
| 7453 | intel | 23,588 | 40 | 4 | 8x8x1 | `cce375e0bde2…` | `aa379f63cf26…` |
| 7454 | intel | 36,012 | 48 | 5 | 8x8x1 | `445267b55959…` | `aac1354a8004…` |
| 7455 | intel | 148,628 | 16 | 5 | 4x16x2 | `9ab994d062aa…` | `aae5e2f67110…` |
| 7456 | intel | 8,516 | 40 | 3 | 8x8x1 | `429d3b28bdba…` | `b2f8fd5c6c6c…` |
| 7457 | intel | 1,488 | 24 | 2 | 8x8x1 | `cd499b678b8d…` | `b5b5d5ab92da…` |
| 7458 | intel | 55,596 | 48 | 6 | 8x8x1 | `f8d3bdbb5365…` | `ba5c8b9ea1aa…` |
| 7459 | port | 154,040 | 16 | 5 | 4x8x4 | `04497edd0b3f…` | `bc9d3fff9f95…` |
| 7460 | intel | 17,448 | 48 | 5 | 8x8x1 | `138943f48549…` | `bca124cb664a…` |
| 7461 | port | 7,788 | 24 | 3 | 8x8x1 | `ec5c0342ff7a…` | `bd0ea2581c98…` |
| 7462 | intel | 8,056 | 56 | 2 | 8x8x1 | `e402d3879337…` | `c0481f4f9956…` |
| 7463 | intel | 8,456 | 40 | 3 | 8x8x1 | `3b5cbca2558f…` | `c68b38430a94…` |
| 7464 | intel | 60,412 | 48 | 6 | 8x8x1 | `7a940640e31d…` | `c714a2a82137…` |
| 7465 | port | 14,124 | 36 | 5 | 8x8x1 | `7181124d275d…` | `c8b778ce1332…` |
| 7466 | intel | 65,540 | 48 | 5 | 8x8x1 | `d92f6d188076…` | `cb00e33e2ae8…` |
| 7467 | intel | 5,752 | 16 | 3 | 8x8x1 | `8a2305ac92b4…` | `cd506291d980…` |
| 7468 | intel | 15,128 | 72 | 13 | 8x8x1 | `f8600efe55f2…` | `ced2990ca1b3…` |
| 7469 | intel | 21,132 | 40 | 4 | 8x8x1 | `b867798c4eed…` | `da17607a0848…` |
| 7470 | intel | 10,456 | 52 | 3 | 8x8x1 | `5f7ce3970001…` | `e419eafc264a…` |
| 7471 | port | 65,540 | 48 | 5 | 8x8x1 | `3a880f707ff4…` | `e6800a53087c…` |
| 7472 | intel | 212,772 | 204 | 17 | 8x16x1 | `e6ffacfb74e0…` | `e71100b0d402…` |
| 7473 | port | 36,012 | 48 | 5 | 8x8x1 | `07572374252f…` | `ea67bb193e1f…` |
| 7474 | intel | 8,056 | 56 | 2 | 8x8x1 | `2648bfd51226…` | `efc1536e7b04…` |
| 7475 | port | 15,236 | 72 | 14 | 8x8x1 | `668fd662896a…` | `f4aa535a6fc7…` |
| 7476 | port | 40,664 | 48 | 5 | 8x8x1 | `5ed9b9f9b2c2…` | `f4ce38ed177a…` |
| 7477 | port | 7,828 | 52 | 2 | 8x8x1 | `b045ea97cbf7…` | `f5326c8b3443…` |
| 7478 | intel | 212,452 | 204 | 16 | 8x16x1 | `8c196e4a9b77…` | `f84a3eb7ab81…` |
| 7479 | intel | 74,064 | 48 | 5 | 8x8x1 | `07de603584ff…` | `fc68cb621ecb…` |
| 7480 | intel | 4,084 | 24 | 2 | 8x8x1 | `68d9d859f814…` | `fd824d2d42bd…` |
| 7481 | intel | 10,096 | 56 | 2 | 8x8x1 | `a05f7e0a9d1d…` | `fea66ef764b2…` |
| 7482 | unhashed | 162,224 | 16 | 5 | 8x8x4 | `4ac28a06e388…` | - |
