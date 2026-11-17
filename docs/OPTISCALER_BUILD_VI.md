# Build trên GitHub và kiểm tra OptiScaler

## Kết luận tương thích

Đã đối chiếu mã nguồn OptiScaler master tại commit
`03b37db73979e042bce4be59f44d37f348ac6905` ngày 17/11/2026.
**Tương thích giao diện plugin/custom DLSSG ở mức mã nguồn; chưa xác nhận chạy
thực tế trong OptiScaler hoặc game Vulkan.** Không có DLL Windows đã build trong
bộ source này. API tương thích không đồng nghĩa driver/game tương thích.

Bằng chứng:

- `plugin_sdk/optiscaler_plugin.h`: SDK v3 bổ sung ở cuối bảng; cả 32 trường đầu
  của mod v1 khớp thứ tự và khai báo. Callback và enum log cũng khớp.
- `OptiScaler/misc/CustomProviders.h::DllPath/Load`: quét
  `custom/<name>/dlssg`, ưu tiên `nvngx_dlssg.dll`, sau đó chọn một DLL trong thư mục,
  và gọi `OptiPluginInit`. Đừng đặt nhiều DLL provider trong cùng thư mục.
- `OptiScaler/framegen/nvngx/Nvngx_Custom.cpp`: dùng custom DLSSG exports;
  provider báo `FrameGeneration.Available=0` thì không được sử dụng.
- `OptiScaler/proxies/Nvngx_DllProxy.cpp`: chuyển các lời gọi NGX Vulkan tới DLL.
- `OptiScaler/framegen/nvngx/Nvngx_FG.cpp`: bọc handle rồi gọi Evaluate của provider.
- `src/Plugin.cpp::FindMainDll/StartFromAsi`: bản `.asi` phải có cùng basename
  với DLL chính; không đặt suffix `-loadlate` cho Vulkan early loader.

Nguồn cố định để xem lại:
https://github.com/optiscaler/OptiScaler/tree/03b37db73979e042bce4be59f44d37f348ac6905

**Điểm chưa thể xác nhận:** runtime mod hiện giả định ảnh NGX ở GENERAL.
Đường HUD-cutoff của OptiScaler có nhắc các layout SHADER_READ_ONLY_OPTIMAL /
TRANSFER_DST_OPTIMAL và tracker riêng; việc forwarding NGX không tự chứng minh
mọi resource đã ở GENERAL. Chưa có thử GPU để chứng minh tương thích layout này.
Depth view chỉ một aspect, queue family/ownership, device recreation và early
hook ordering vẫn là các điều kiện cần kiểm tra trong game. Bản sửa không bổ sung
swapchain/presentation của OptiScaler cho một game chưa có đường FG input phù hợp.

## File cần đưa vào GitHub

Nếu dùng `changed-files.zip`: giải nén **đè tại root repository nguồn ban đầu**,
ngang hàng `dlssg-to-xefg.sln`. ZIP này chỉ chứa file mới/sửa, không phải repository
hoàn chỉnh. Không giải nén thành một thư mục con thừa.

- Workflow: `.github/workflows/build.yml` (bản thay thế, không cần chạy song song bản cũ).
- Payload: giải nén `vulkan-payload-for-repo.zip` tại cùng root. Nó tạo
  `third_party/xefg_mfg_programs/vulkan/` chứa shader đã gửi và đã kiểm chứng.
- Nếu dùng `source-ready-for-github.zip`: đã có đủ source và payload trên, cấu trúc
  root sẵn dùng, bao gồm `.github` và action lấy Vulkan headers.
- `third_party/libxess_fg.dll` là tùy chọn để kiểm tra và kèm vào artifact;
  nếu không đưa vào repo thì tự cài DLL Intel tương thích sau khi tải artifact.
  `.gitignore` gốc bỏ qua `*.dll`; nếu chủ động đưa DLL vào repo bằng git,
  cần `git add -f third_party/libxess_fg.dll`.

Bật Actions trên repository rồi vào **Actions → Build OptiScaler XeMFG Vulkan →
Run workflow**. Workflow cũng chạy khi push hoặc pull request.

## Workflow làm gì

1. Chạy Linux provider tests, policy/regression tests, syntax checks và đọc payload thật.
2. Đối chiếu prefix SDK OptiScaler với commit cố định nói trên.
3. Windows x64 Release chỉ build sau khi cả ba job kiểm tra đạt; lấy Vulkan-Headers
   qua `.github/actions/vulkan-headers`, dùng MSBuild và xác nhận log
   `Vulkan runtime: built in`.
4. Generator hiện có tải **bản MFG-Unlocker đã pin hash** để trích DXIL + CUDA fatbin.
   Không cần tự cài CUDA toolkit nếu dùng fatbin đó. Download/hash sai làm build thất bại.
5. Bắt buộc có payload Vulkan; không xuất gói "thành công" mà thiếu shader.
6. Kiểm tra exports trong PE, dùng Windows loader nạp DLL và tìm exports.
   Smoke test không gọi OptiPluginInit/InitializeASI/NGX, không chạy GPU.
7. Xuất artifact `dlssg-to-xefg-<run number>` đúng cây thư mục cài đặt bên dưới.
   Nếu lỗi build, tìm thêm artifact `msbuild-log-<run number>` và log từng step.

Đã kiểm tra tại đây: YAML parse, **actionlint v1.7.7 đạt**, ABI prefix 32 trường đạt,
negative test ABI đổi chữ ký bị từ chối, **77 + 221 provider tests đạt** trên Linux
với ASan/UBSan. Payload 83 shader đã qua 280 kiểm tra ở lượt trước và được copy
nguyên vẹn. Chưa chạy Actions trên repository của bạn; bước Windows build/smoke
và kết quả thử game vẫn chưa có.

## Cài artifact

Giải nén artifact vào **thư mục OptiScaler thực tế có `OptiScaler.ini` và `custom`**:

```text
<OptiScaler root>/
  custom/
    XeMFG XMX/
      dlssg/
        dlssg_to_xefg_intel_is_better.dll
        xefg_mfg_programs/
          programs.tsv               # DXIL, do generator tạo khi build
          ... .dxil
          xefg_cuda_layers.fatbin
          vulkan/
            programs.tsv
            7400.spv ... 7482.spv
            input_copy_0.spv ...
  plugins/
    dlssg_to_xefg_intel_is_better.asi
  libxess_fg.dll                       # cần bản tương thích; có thể tự chép vào
  README-VI.md
```

Giữ nguyên tên DLL và ASI; dùng hai file cùng một artifact. Không đổi DLL thành
`nvngx_dlssg.dll` nếu vẫn dùng early loader với basename hiện tại. Giữ `dlssg` chỉ
có một DLL provider. Đặt `libxess_fg.dll` ở root OptiScaler, như artifact làm khi
repo có DLL Intel: nếu đặt nó cạnh provider tên không chuẩn thì DllPath có thể
chọn nhầm DLL đầu tiên. Runtime tìm lên tối đa ba cấp để thấy DLL Intel ở root.

Trong `OptiScaler.ini`, gộp vào section sẵn có, không tạo các section trùng:

```ini
[Plugins]
LoadAsiPlugins = true

[Plugin.XeMFG XMX]
VulkanHooks = true
```

Dùng bản OptiScaler có custom providers/plugin SDK tương ứng. Chọn
**XeMFG XMX** làm DLSSG replacement/FG output trong menu theo FG input của game;
Save Settings rồi **khởi động lại game** để early loader chạy trước lúc tạo device.
Tên vị trí menu có thể khác giữa các bản OptiScaler.

Log mod:
`custom/XeMFG XMX/dlssg/dlssg_to_xefg_intel_is_better.log`.
Cần thấy hooks được cài, device được chuẩn bị, bộ Vulkan có 83 programs và context
XeFG được tạo. `device_not_prepared`/`instance_unknown` nghĩa là chưa nạp sớm đủ.
`no_vulkan_models` là thiếu/sai payload. `feature_chain_unsupported` là cấu trúc
pNext chưa thể sao chép an toàn. Hiện tab plugin không chứng minh FG đã chạy.

Thử XMX Port/Fallback trước, sau đó CUDA; đối chiếu log model thực tế.
Kiểm tra alt-tab, đổi resolution, pause/resume, HUD, HDR và nhiều mức nhân khung.
XMX Port chạy SPIR-V, không phải bảo đảm dùng XMX phần cứng Intel. Không bật các
mod FG khác đồng thời trong lần kiểm chứng đầu tiên.
