#include "vulkan/VulkanFeatureChain.h"
#include <cassert>
#include <cstdio>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif
int main()
{
    using VulkanHost::FeatureChain;
    VkBaseInStructure tail{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &tail};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    FeatureChain chain;
    assert(chain.Copy(&v12));
    auto* copy12 = static_cast<VkPhysicalDeviceVulkan12Features*>(const_cast<void*>(chain.Head()));
    auto* copy13 = static_cast<VkPhysicalDeviceVulkan13Features*>(copy12->pNext);
    assert(copy12 != &v12 && copy13 != &v13 && copy13->pNext == &tail);
    copy12->timelineSemaphore = VK_TRUE;
    copy13->maintenance4 = VK_TRUE;
    assert(v12.timelineSemaphore == VK_FALSE && v13.maintenance4 == VK_FALSE);
    VkBaseInStructure unknown{VK_STRUCTURE_TYPE_APPLICATION_INFO, reinterpret_cast<VkBaseInStructure*>(&v12)};
    assert(!chain.Copy(&unknown)); // cannot safely copy an unknown prefix
    v12.timelineSemaphore = VK_TRUE;
    v13.synchronization2 = v13.maintenance4 = v13.shaderIntegerDotProduct = VK_TRUE;
    assert(chain.Copy(&unknown) && chain.Head() == &unknown); // no edit: preserve arbitrary extensions
    assert(chain.Copy(nullptr) && !chain.Head());
    VkBaseInStructure cycle{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr};
    cycle.pNext = &cycle;
    assert(!chain.Copy(&cycle));
#ifdef __linux__
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* storage = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(storage != MAP_FAILED);
    auto* readonly = new (storage) VkPhysicalDeviceVulkan13Features{};
    readonly->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    assert(mprotect(storage, page, PROT_READ) == 0);
    assert(chain.Copy(readonly));
    auto* writable = static_cast<VkPhysicalDeviceVulkan13Features*>(const_cast<void*>(chain.Head()));
    writable->synchronization2 = VK_TRUE;
    assert(readonly->synchronization2 == VK_FALSE);
    assert(munmap(storage, page) == 0);
#endif
    std::puts("feature chain: immutable input, copied prefix, borrowed tail, unknown nodes, cycle: PASS");
}
