// Executes the runtime's actual copy helper and EnsureFeature init-failure suppression (including Path::Wmma).
// Other runtime entry points are discarded by --gc-sections; no GPU is used.
#include "../../src/vulkan/VulkanRuntime.cpp"
#include <cassert>
#include <cstdio>

static VkImageCopy captured{};
static VkImage capturedSource{}, capturedTarget{};
static void VKAPI_CALL Capture(VkCommandBuffer, VkImage source, VkImageLayout, VkImage target,
    VkImageLayout, uint32_t count, const VkImageCopy* regions)
{
    assert(count == 1);
    captured = *regions;
    capturedSource = source;
    capturedTarget = target;
}

bool Log::IsEnabled() noexcept { return true; }
void Log::Write(std::string_view, std::string_view) {}

static uint64_t gFakeMissingProgramCount = 0;
static bool gSimulateMissingProgramOnInit = false;

namespace xvk
{
    void SetTensorCoresEnabled(ID3D12Device*, bool) noexcept {}
    void SetCoopMatrixEnabled(ID3D12Device*, bool) noexcept {}
    uint64_t MissingProgramCount() noexcept { return gFakeMissingProgramCount; }
    VkImageView StorageImageView(ID3D12Resource*, VkFormat) noexcept
    {
        return reinterpret_cast<VkImageView>(0x100);
    }
    VkBuffer BufferOf(ID3D12Resource*) noexcept { return VK_NULL_HANDLE; }
}

static int gCreateContextCalls = 0;
static int gInitCalls = 0;
static int gDestroyCalls = 0;
static int32_t gLastOverriddenModel = -1;
static xefgd_result_t gCreateContextResult = XEFGD_RESULT_SUCCESS;
static xefgd_result_t gInitResult = XEFGD_RESULT_SUCCESS;

namespace XeFG
{
    void SetMaxInterpolatedFrames(uint32_t) {}
    void OverrideModel(int32_t model) { gLastOverriddenModel = model; }
    xefgd_result_t CreateContext(ID3D12Device*, xefgd_handle_t* out)
    {
        ++gCreateContextCalls;
        if (gCreateContextResult == XEFGD_RESULT_SUCCESS && out)
            *out = reinterpret_cast<xefgd_handle_t>(0xCAFE);
        return gCreateContextResult;
    }
    void SetLoggingCallback(xefgd_handle_t, xefgd_logging_level_t, xefgd_log_callback_t) {}
    xefgd_result_t Init(xefgd_handle_t, const xefgd_init_params_t&)
    {
        ++gInitCalls;
        if (gSimulateMissingProgramOnInit)
            ++gFakeMissingProgramCount;
        return gInitResult;
    }
    xefgd_result_t Destroy(xefgd_handle_t)
    {
        ++gDestroyCalls;
        return XEFGD_RESULT_SUCCESS;
    }
}

namespace
{
    struct FakeResource final : ID3D12Resource
    {
        unsigned long refs = 1;
        unsigned long AddRef() override { return ++refs; }
        unsigned long Release() override
        {
            const auto r = --refs;
            if (!r) delete this;
            return r;
        }
        HRESULT QueryInterface(REFIID, void** out) override
        {
            if (!out) return E_INVALIDARG;
            *out = this;
            ++refs;
            return S_OK;
        }
    };

    struct FakeAllocator final : ID3D12CommandAllocator
    {
        unsigned long refs = 1;
        unsigned long AddRef() override { return ++refs; }
        unsigned long Release() override
        {
            const auto r = --refs;
            if (!r) delete this;
            return r;
        }
        HRESULT QueryInterface(REFIID, void** out) override
        {
            if (!out) return E_INVALIDARG;
            *out = this;
            ++refs;
            return S_OK;
        }
        HRESULT Reset() override { return S_OK; }
    };

    struct FakeDevice final : ID3D12Device
    {
        unsigned long refs = 1;
        int resourceAllocations = 0;
        unsigned long AddRef() override { return ++refs; }
        unsigned long Release() override { return --refs; }
        HRESULT QueryInterface(REFIID, void** out) override
        {
            if (!out) return E_INVALIDARG;
            *out = this;
            ++refs;
            return S_OK;
        }
        HRESULT CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE, REFIID, void** allocator) override
        {
            *allocator = new FakeAllocator();
            return S_OK;
        }
        HRESULT CreateCommandList(UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*, void*, REFIID, void**) override
        {
            return S_OK;
        }
        HRESULT CreateCommittedResource(const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
            D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void** resource) override
        {
            ++resourceAllocations;
            *resource = new FakeResource();
            return S_OK;
        }
    };

    void TestEnsureFeatureWmmaInitFailureSuppression()
    {
        FakeDevice fakeD3d;
        gDevice.d3d = &fakeD3d;
        gDevice.portInstalled = true;
        gDevice.coopInstalled = true;
        gDevice.tensorInstalled = false;
        gDevice.vendor = 0x1002; // AMD RDNA 4 / RDNA 3

        Plugin::options.path.store(Plugin::Path::Wmma);

        NgxResource bb{}, depth{}, motion{};
        bb.image = reinterpret_cast<VkImage>(10);
        bb.format = VK_FORMAT_R8G8B8A8_UNORM;
        bb.width = 1920;
        bb.height = 1080;
        bb.range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        depth.image = reinterpret_cast<VkImage>(11);
        depth.format = VK_FORMAT_R32_SFLOAT;
        depth.width = 1920;
        depth.height = 1080;
        depth.range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        motion.image = reinterpret_cast<VkImage>(12);
        motion.format = VK_FORMAT_R16G16_SFLOAT;
        motion.width = 1920;
        motion.height = 1080;
        motion.range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        Inputs in{ &bb, &depth, &motion, nullptr, nullptr };
        Feature f{};

        // 1. Simulate XeFG::Init failure under Path::Wmma across 5 consecutive frames:
        //    must call CreateContext/Init only ONCE and suppress repeated init failures on frames 2..5.
        gCreateContextCalls = 0;
        gInitCalls = 0;
        gDestroyCalls = 0;
        gCreateContextResult = XEFGD_RESULT_SUCCESS;
        gInitResult = static_cast<xefgd_result_t>(-1);

        for (int frame = 0; frame < 5; ++frame)
        {
            const bool ok = EnsureFeature(gDevice, f, in, true, true);
            assert(!ok);
            assert(f.createFailed);
            assert(f.context == nullptr);
            assert(f.fault == "init_-1");
        }
        assert(gCreateContextCalls == 1);
        assert(gInitCalls == 1);
        assert(gDestroyCalls == 1);
        assert(gCoopGate.load() == true);
        assert(gLastOverriddenModel == MfgKernels::MfgModel);
        const int allocsAfterFirstTry = fakeD3d.resourceAllocations;
        assert(allocsAfterFirstTry == 6 + static_cast<int>(MaxGenerated)); // 2 slots * 3 inputs + 6 outputs, allocated only on frame 0

        // 2. Switching network path (e.g., Path::XmxPort -> Path::Wmma) resets createFailed and retries once.
        gInitResult = XEFGD_RESULT_SUCCESS;
        Plugin::options.path.store(Plugin::Path::XmxPort);
        assert(EnsureFeature(gDevice, f, in, true, true));
        assert(!f.createFailed);
        assert(f.context != nullptr);
        assert(gCreateContextCalls == 2 && gInitCalls == 2);

        // 3. Subsequent frames on Path::Wmma reuse the live context without re-initializing.
        Plugin::options.path.store(Plugin::Path::Wmma);
        assert(EnsureFeature(gDevice, f, in, true, true));
        assert(EnsureFeature(gDevice, f, in, true, true));
        assert(gCreateContextCalls == 2 && gInitCalls == 2);
        assert(std::string(ModelText(f.model)) == "WMMA");

        // 4. Missing-SPIR-V during XeFG::Init latches initFault ("init_-1000_missing_spirv") per input configuration,
        //    even if the user toggles paths that keep the same model (e.g., Path::Wmma <-> Path::XmxPort), and clears
        //    only when the input configuration or model changes.
        Feature fMissing{};
        gCreateContextCalls = 0;
        gInitCalls = 0;
        gDestroyCalls = 0;
        gInitResult = static_cast<xefgd_result_t>(-1000);
        gSimulateMissingProgramOnInit = true;
        Plugin::options.path.store(Plugin::Path::Wmma);

        assert(!EnsureFeature(gDevice, fMissing, in, true, true));
        assert(fMissing.initFault == "init_-1000_missing_spirv");
        assert(fMissing.fault == "init_-1000_missing_spirv");
        assert(gCreateContextCalls == 1 && gInitCalls == 1 && gDestroyCalls == 1);

        // Switching between Wmma and XmxPort keeps model == 16 (same input config), so initFault remains latched:
        Plugin::options.path.store(Plugin::Path::XmxPort);
        assert(!EnsureFeature(gDevice, fMissing, in, true, true));
        assert(fMissing.fault == "init_-1000_missing_spirv");
        assert(gCreateContextCalls == 1 && gInitCalls == 1);

        // Changing resolution (input configuration change) clears initFault and retries cleanly:
        gSimulateMissingProgramOnInit = false;
        gInitResult = XEFGD_RESULT_SUCCESS;
        Plugin::options.path.store(Plugin::Path::Wmma);
        bb.width = depth.width = motion.width = 2560;
        bb.height = depth.height = motion.height = 1440;
        assert(EnsureFeature(gDevice, fMissing, in, true, true));
        assert(fMissing.initFault.empty());
        assert(fMissing.context != nullptr);
        assert(gCreateContextCalls == 2 && gInitCalls == 2);

        gDevice.d3d.Reset();
    }
}

int main()
{
    gDevice.vkCmdCopyImage = Capture;
    NgxResource source{}, target{};
    source.image = reinterpret_cast<VkImage>(1);
    target.image = reinterpret_cast<VkImage>(2);
    source.range = {VK_IMAGE_ASPECT_COLOR_BIT, 3, 1, 2, 1};
    target.range = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 4, 1};
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 640, 360, &source, &target);
    assert(capturedSource == source.image && capturedTarget == target.image);
    assert(captured.srcSubresource.mipLevel == 3 && captured.srcSubresource.baseArrayLayer == 2);
    assert(captured.dstSubresource.mipLevel == 1 && captured.dstSubresource.baseArrayLayer == 4);
    assert(captured.extent.width == 640 && captured.extent.height == 360 && captured.extent.depth == 1);
    assert(captured.srcSubresource.layerCount == 1 && captured.dstSubresource.layerCount == 1);
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 128, 64, nullptr, &target);
    assert(captured.srcSubresource.mipLevel == 0 && captured.srcSubresource.baseArrayLayer == 0);
    assert(captured.dstSubresource.mipLevel == 1 && captured.dstSubresource.baseArrayLayer == 4);
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 128, 64, &source);
    assert(captured.srcSubresource.mipLevel == 3 && captured.dstSubresource.mipLevel == 0);
    assert(!SameSubresource(source, target));
    target.image = source.image;
    assert(!SameSubresource(source, target));
    target.range = source.range;
    assert(SameSubresource(source, target));

    TestEnsureFeatureWmmaInitFailureSuppression();
    std::puts("runtime copies & WMMA init-failure suppression: PASS");
}
