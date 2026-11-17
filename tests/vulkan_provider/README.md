# Vulkan provider tests

Run the real `src/VulkanProvider.cpp` and `src/ExportsVulkan.cpp` on Linux (or anywhere with g++ 13 or newer) against a fake
process. `stubs/` holds just enough of the Win32 API for that. They check this dll's side of the 12 `NVSDK_NGX_VULKAN_*`
exports in the two ways the exports find their provider, and say nothing about a game or about the real runtime:

| Test | Provider | What it covers |
|---|---|---|
| `vk_builtin_test.cpp` | the runtime built into the dll (`src/vulkan/VulkanRuntime.cpp`), here a fake that answers at its table like the real one | nothing is searched for, `Init` starts the hooks and hands the device over, the runtime refusing a device, a device that can't run XeMFG (refused at once, said once, shown in the menu), what keeps it from running before a game gave it a device, the handles, the parameters and the result codes of all 12 exports |
| `vk_test.cpp` | a build without the Vulkan headers (`src/VulkanRuntimeOff.cpp`) and a stand-in for the provider table of `MFGUnlocker.asi` 1.7.1-beta2fix.1 | finding the provider, the table check, `Init` refusing when the asi says it can't run, the handles, the parameters and the result codes of all 12 exports |

```
g++ -std=c++20 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-cast-function-type \
    -fsanitize=address,undefined -g -D'__declspec(x)=' -I stubs -I ../../src \
    vk_builtin_test.cpp ../../src/ExportsVulkan.cpp ../../src/VulkanProvider.cpp -o vk_builtin_test && ./vk_builtin_test

g++ -std=c++20 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-cast-function-type \
    -fsanitize=address,undefined -g -D'__declspec(x)=' -I stubs -I ../../src \
    vk_test.cpp ../../src/ExportsVulkan.cpp ../../src/VulkanProvider.cpp ../../src/VulkanRuntimeOff.cpp -o vk_test && ./vk_test
```

They print `77 checks, 0 failed` and `221 checks, 0 failed` and exit 0 when everything holds. `build.yml` runs both.
