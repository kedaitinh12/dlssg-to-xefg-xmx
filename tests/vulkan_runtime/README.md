# Vulkan runtime checks

The Vulkan runtime built into the dll (`src/vulkan/`) only builds on Windows with the Vulkan headers. These checks cover what
can be checked without either, and say nothing about how it runs in a game or on a GPU.

| Check | What it does |
|---|---|
| `policy_test.cpp` | The decisions in `src/vulkan/VulkanPolicy.h` that need neither Vulkan nor Windows: when a batch of generated frames starts, which outputs an evaluation records, how the game's formats and camera map to what XeFG reads, what each reason code means, and for the tensor-core route the words of its environment switches (`XEFG_TENSOR_CORES`, `XEFG_TENSOR_CORES_VULKAN_BARRIERS`), the parsing of `XEFG_TENSOR_CORES_MASK`, which network paths use it, and the texts of its reasons. The expected values are written out in the test from MFG-Unlocker's own policy and from the formats' registry values, not computed by the header's code. 365 checks. |
| `check_reasons.py` | Every reason code the runtime and its hooks can give (string literals in `VulkanHost.cpp` and `VulkanRuntime.cpp`) has a text in `VulkanPolicy::Describe`, and every code of the tensor-core route (`xvk_tensor.cpp`, `VulkanTensor.cpp`, `InstallTensorRoute` in `VulkanRuntime.cpp`) one in `VulkanPolicy::DescribeTensor`. No text is there twice, and none is there that nothing in the sources produces, except `no_hooks` and `no_loader`, which name states the menu shows. |
| `check_layer_tables.py` | The tensor-core route's layer table is in the repository twice: inline in `src/MfgPort/Cuda/cuda_route.cpp` (D3D12) and in `src/MfgPort/Cuda/cuda_layers.h`, MFG-Unlocker's header that the Vulkan route reads. The two have to name the same 16 kernels, the same 17 programs with the same SHA-256, the same shapes. With `--fatbin FILE` it also checks that the file is the pinned `xefg_cuda_layers.fatbin` (SHA-256 `2ffae52e…`, `--no-pin` accepts another) and holds every kernel by name, which is what the route checks before it imports the module. Needs no GPU. |
| `check_syntax.sh` | A syntax check, no more, of the Windows-only sources (`VulkanHost.cpp`, `VulkanRuntime.cpp`, `VulkanTensor.cpp`, `xvk_extra.h`, `xvk_tensor.h`, the tensor-core route's `cuda_route.h`, `cuda_layers.h` and `cuda_vulkan.cpp`, `VulkanProvider.cpp`, `ExportsVulkan.cpp`, `Plugin.cpp`) with `clang++` and `g++`, against `stubs/`: the part of the Win32, D3D12, WRL and Vulkan APIs they use (`VK_NVX_binary_import` and `VK_NVX_image_view_handle` included), with the real names, member order and numeric values, and the real MinHook header. It catches typos, names that aren't declared, calls that don't match a declaration and the `static_assert`s on the format values. It can't tell that a stub is the real API, the Windows build with the real headers is the check of that. |
| `tensor_api_check.cpp` | Part of `check_syntax.sh`: the calls `src/vulkan/xvk_tensor.cpp` makes into `cuda_route.h` and `cuda_layers.h` (`AdapterCapability`, `ReportVulkan`, `CountVulkan`, `LayerEnabled`, `DispatchShapeOk`, `FillLaunch`, ...), written out so that the declarations this repository added under MFG-Unlocker's file are proved to match. |

```
g++ -std=c++20 -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -I ../../src policy_test.cpp -o policy_test && ./policy_test
python3 check_reasons.py
python3 check_layer_tables.py [--fatbin ../../bin/Release/xefg_mfg_programs/xefg_cuda_layers.fatbin]
bash check_syntax.sh         # every one of clang++ and g++ it finds, or name the compilers: bash check_syntax.sh g++
```

The patched files of MFG-Unlocker's layer (`xvk_device.cpp`, `xvk_programs.cpp`, `xvk_util.cpp`, `xvk_tensor.cpp`) and the rest of
`xvk_*.cpp` aren't covered: they take the whole D3D12 and Vulkan API. `tools/VendorXvk.py --check <MFG-Unlocker checkout>`
compares them with MFG-Unlocker's files instead, see `src/vulkan/VENDORED.txt`.
