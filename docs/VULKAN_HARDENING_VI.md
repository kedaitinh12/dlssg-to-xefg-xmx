# Đối chiếu Vulkan và bản sửa runtime

Phạm vi: ba ZIP do người dùng cung cấp. Bản sửa này tăng độ chặt chẽ của runtime
NGX/Vulkan dùng chung cho CUDA, XMX Port và Fallback; không thay trọng số model,
không sửa bytecode SPIR-V và chưa chứng nhận game nào chạy ổn định.

## Kết luận từ đối chiếu

`dlssg-to-xefg-xmx-master` đã có D3D12-on-Vulkan (xvk), NGX Vulkan provider,
hooks tạo instance/device, input-copy shader và đường CUDA qua NVX.
Không cần chép toàn bộ Vulkan của RenoDx sang để bắt đầu hỗ trợ Vulkan.
Các file `xvk_commandlist.cpp`, `xvk_resource.cpp`, `xvk_pipeline.cpp`,
`xvk_queue.cpp` giống byte-for-byte bản vendor trong ZIP RenoDx.
`xvk_tensor.cpp` chỉ khác đường dẫn include `cuda_layers.h`.
Vì thế khác biệt đáng chú ý nằm ở lớp tích hợp với game và vòng đời tài nguyên,
không phải một bộ CUDA kernel tốt hơn trong RenoDx.

Nguồn đối chiếu trong ZIP RenoDx:

- `src/addons/mfgunlock/xemfg/native_vulkan.cpp`: hooks theo dõi image/device,
  tạo sampled input view, xóa view khi image/device bị hủy.
- `src/addons/mfgunlock/xemfg/vulkan_policy.hpp`: `ChooseInputView`,
  `DispatchKey`, `DisallowedChain`, chính sách present.
- `src/addons/mfgunlock/xemfg/vendor/xvkhost/vk_provider.cpp` và `vendor/xvk/`:
  provider và lớp chuyển lệnh.

| Hạng mục | Đối chiếu / vấn đề | Bản sửa này |
|---|---|---|
| Shader và CUDA | Bộ gửi kèm có 83 SPIR-V, cùng 17 lớp/16 CUDA kernel đã khai báo | Kiểm tra hash/manifest; giữ nguyên payload |
| `pNext` thiết bị | Mã cũ sửa trực tiếp feature trong cấu trúc `const` của game, cả sau khi thành công | Sao chép prefix cần sửa; giữ tail nguyên vẹn; không ghi vào bộ nhớ game |
| Cấu trúc chưa biết | Không thể đoán kích thước một Vulkan extension structure | Nếu phải sao chép qua node chưa hỗ trợ: không bật FG, giữ yêu cầu tạo device gốc, báo `feature_chain_unsupported` |
| Command buffer | RenoDx dùng dispatch key để phân biệt device | Kiểm tra key trước khi ghi lệnh lên device NGX; đây không phải hỗ trợ multi-device |
| Color copy | Mã cũ luôn dùng mip 0/layer 0 | Dùng `baseMipLevel`/`baseArrayLayer` từ NGX cho cả input/output, nhận diện alias theo subresource |
| Format output | Đường generated-output chưa kiểm tra format đích | Chỉ chấp nhận format có cùng ánh xạ màu, gồm cặp UNORM/sRGB tương ứng; không coi RGBA/BGRA hoặc HDR packed/FP16 là tương đương |
| HUDless/UI | Allocation theo size riêng nhưng XeFG nhận extent của backbuffer; resize có thể dùng lại allocation sai | Bỏ qua ảnh UI/HUDless khác kích thước backbuffer; vẫn sinh khung với input còn hợp lệ, không resample UI |
| Depth/motion view | RenoDx có bảng metadata image và tự tạo view một aspect | Bản này kiểm tra view/aspect; từ chối depth+stencil view hoặc view rỗng, chưa tự sửa view |
| Reset / Execute thất bại | Lịch sử có thể được đánh dấu hợp lệ trước Execute thành công | Chỉ chuyển slot sau thành công; lỗi batch/output/exception xóa trạng thái dùng lịch sử; frame đầu hoặc khôi phục đặt resetHistory |
| OutputReal | Có thể đọc backbuffer đã bị output nội suy ghi đè | Ghi real trước generated; khi có batch hợp lệ lấy real từ bản input đã lưu |
| Present / swapchain | RenoDx có ReShade/Streamline, private data, present-mode và FSE hooks | Không ghép vào NGX provider: host/game vẫn sở hữu submit/present |

## Ý nghĩa ba đường chạy trên Vulkan

| Tùy chọn | Thực thi |
|---|---|
| CUDA | Model port; các lớp phù hợp chạy kernel NVX trên NVIDIA hỗ trợ; lớp không phù hợp giữ SPIR-V |
| XMX Port | Model port được thực thi bằng SPIR-V; không đồng nghĩa dùng XMX phần cứng Intel |
| XMX (native) | Hành vi hiện có trên Vulkan: tương tự lựa chọn CUDA nếu có tensor route, nếu không dùng SPIR-V |
| Fallback | Model DP4a của XeFG bằng SPIR-V; không dùng các lớp CUDA của model port |

Bản sửa áp dụng chung cho việc nạp/copy input, kiểm tra output và lịch sử của tất cả
các đường trên. Không thay kiến trúc kernel, thuật toán motion estimation hoặc
lịch submit/present. Intel GPU vẫn bị runtime gốc từ chối trên Vulkan.

## Đã kiểm chứng

Trên Linux, GCC 14.2:

- Policy: **377 checks, 0 failed**, ASan/UBSan.
- Đọc bộ shader thật người dùng gửi: **280 checks, 0 failed**;
  **83 programs** (port 25, port-extra 1, Intel 56, unhashed 1).
- Bộ synthetic/malformed và missing payload: **206 + 5 checks, 0 failed**.
- Hai bảng CUDA khớp **17 layers, 16 kernels**. Fatbin 1.122.728 byte,
  SHA-256 `2ffae52e7878ae7e374ff953b864204f079c3199dc904b2ce7ecf8b37ee130f7`.
- Kiểm tra reason codes đạt.
- Syntax-check 12 source/header targets bằng stubs Win32/D3D12/Vulkan đạt.
- Regression `feature_chain_test`: copied prefix, borrowed tail, unknown prefix,
  cycle, tái sử dụng object và trang chỉ đọc qua `mprotect` đạt, ASan/UBSan.
  Cũng đạt khi dùng Vulkan-Headers **v1.3.290 thật** cho riêng test này.
- Regression `copy_test`: gọi helper CopyImage thật, bắt lệnh vkCmdCopyImage
  để kiểm tra mip/layer nguồn-đích, extent, internal texture và alias; đạt, UBSan.
- Hai regression mới được thêm vào workflow CI.

**Chưa build/link DLL Windows, chưa chạy libxess_fg.dll, chưa chạy CUDA/SPIR-V
trên GPU và chưa đo FPS/latency hoặc xác minh hình ảnh trong game.**
Syntax-check với stubs không thay thế build bằng Windows SDK và Vulkan headers thật.

Chạy lại:

```bash
bash tests/vulkan_runtime/check_regressions.sh
bash tests/vulkan_runtime/check_syntax.sh
python tests/vulkan_runtime/check_reasons.py
python tests/vulkan_runtime/check_layer_tables.py --fatbin /path/to/xefg_cuda_layers.fatbin
```

Các lệnh policy/program reader nằm trong README của từng thư mục tests.

## Build và lắp vào game

1. Build source trên Windows bằng Visual Studio/MSBuild theo README gốc.
   Dùng Vulkan headers phiên bản theo `.github/actions/vulkan-headers/action.yml`;
   truyền `VulkanIncludeDir` để tránh vô tình build bản runtime-off:

   ```powershell
   msbuild dlssg-to-xefg.sln -p:Configuration=Release -p:Platform=x64 -m "-p:VulkanIncludeDir=C:\Vulkan-Headers\include"
   ```

   Log phải có `Vulkan runtime: built in`. Không coi kết quả chỉ có DLL là đủ.
2. Cài DLL theo cấu trúc OptiScaler trong README gốc. Sao chép thư mục
   `runtime-payload/xefg_mfg_programs` của gói kết quả cạnh DLL, giữ nguyên
   `vulkan/programs.tsv`, 83 chương trình, input-copy shaders và CUDA fatbin.
3. Vẫn cần `libxess_fg.dll` tương thích và bộ DXIL của model port theo README.
   ZIP `xefg_mfg_programs-vulkan` được gửi **chỉ có Vulkan và CUDA fatbin**;
   nó không cung cấp DLL Intel hoặc toàn bộ DXIL của model port. Đủ SPIR-V không
   có nghĩa MfgKernels::Install sẽ thành công. Nếu thiếu port, runtime có thể chọn DP4a.
4. Nạp hooks trước khi game tạo instance/device. Thông báo `device_not_prepared`
   không thể chữa chỉ bằng đổi model sau khi device đã được tạo; xem Early start
   trong README và khởi động lại game khi thay cách nạp.
5. Với DLL Intel thực tế, chạy:

   ```powershell
   python tools/CheckXeFgBuild.py C:\path\libxess_fg.dll --vulkan-programs C:\path\xefg_mfg_programs\vulkan
   ```

## Ma trận cần thử trước khi coi là ổn định

Chạy cùng cảnh/save ở XMX Port, CUDA và Fallback; kiểm tra log để biết model thực tế,
không suy luận từ tên tùy chọn. Với CUDA thử thêm `XEFG_TENSOR_CORES=0` để đối chiếu.

- 2x đến 6x; index liên tiếp và bỏ qua index, pause/resume FG, loading/menu/camera cut.
- Đổi độ phân giải, dynamic resolution, đổi HUDless/UI, đổi model, alt-tab,
  fullscreen/borderless, HDR/SDR và UNORM/sRGB.
- Depth D24S8/D32S8 phải được host cung cấp **sampled view chỉ có depth aspect**;
  kiểm tra đảo depth, jitter và motion scale ở độ phân giải nội bộ khác output.
- Kiểm tra frame thật lẫn frame nội suy, HUD ghosting, texture đen, khung bị giữ lại,
  validation errors, device lost, VRAM theo thời gian và frametime p95/p99.
- NVIDIA cho CUDA/NVX, AMD cho SPIR-V/DP4a; thử driver cụ thể của máy đích.

Những khoảng trống còn lại: theo dõi image format/usage/samples và lifetime như
RenoDx; tự tạo depth view đúng format thực; hỗ trợ device destroy/recreate;
queue-family/queue ownership và synchronization cho async compute/multi-queue;
layout input khác GENERAL; tracking subresource/layout tự động; kiểm tra GPU.
Dispatch key không xác minh queue family. Các barrier hiện tại không tự thực hiện
queue-family ownership transfer. Bản này vẫn yêu cầu host quản lý submit/lifetime
và cung cấp NGX resource đúng hợp đồng, command buffer hợp lệ ngoài render pass,
ảnh single-sample với usage phù hợp. Không tuyên bố đã giải quyết các khoảng trống đó.
