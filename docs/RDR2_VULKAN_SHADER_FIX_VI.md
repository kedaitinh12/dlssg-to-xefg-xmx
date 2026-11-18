# RDR2 Vulkan: sửa bảng shader và vòng lặp khởi tạo FG

Log mới đã tạo được Vulkan feature, nhưng không tạo được context XeFG:
- Thiếu DXIL `00b1098a80a5e662de426284e6bf7bdf82ed236eb7c8aef1e74f9ebed204019a` (key `34811228111448562003997`).
- Thiếu biến thể splat inverted-depth `c3649191906bdd300462dbafe6add8f82b09e8e10c41e92784a209d99513224b`.
- `batch failed (init_-1000)`; 5.240 lỗi pipeline của key đầu tiên, khoảng cách trung vị 41 ms trong các khoảng dưới 1 giây. Đây là nhịp lặp lỗi, không phải phép đo thời gian chạy kernel.

## Nguyên nhân và bản sửa

`ProgramAsset` 64-bit có con trỏ hash DXIL tại offset 0 và resource id tại offset 8. Công cụ cũ bắt đầu giải mã tại resource id, lấy hash của dòng kế tiếp. Hash/size SPIR-V vẫn đúng nên kiểm tra toàn vẹn cũ không phát hiện ánh xạ sai.

Đã trích lại bảng từ MFG-Unlocker 1.7.1 có SHA256 `5a389e6cad23b2b607e1dc224e9e86ee32d6ef46f0557b6c6f8137feadf0aa74`. Cả 86 file SPIR-V giữ nguyên; bảng mới có 83 hash DXIL (26 port, 56 Intel, 1 port-extra), SHA256 manifest `b2e2822b3281c31fc1f71e048e29ce283fc4e83e7f1b65651489ab1d9da5b37e`.

Bổ sung tra cứu hai chiều chỉ cho các cặp shader Intel/thay thế đã được ghim trong source, để Intel inverted-depth splat dùng bản dịch của shader thay thế tương đương. Ưu tiên hash khớp trực tiếp. Loại bỏ cách đoán dòng kế tiếp từ root signature: layout tương thích không chứng minh cùng thuật toán.

Runtime từ chối hai hash manifest lỗi đã biết. Nếu Init vẫn thiếu SPIR-V, ngừng thử lại với cùng cấu hình input để tránh tạo/destroy context và pipeline mỗi frame. Muốn nạp lại payload phải restart game.

## Cài đặt

Dùng artifact mới từ workflow **Build OptiScaler XeMFG Vulkan** trên nhánh `codex/vulkan-hardening-optiscaler`.

1. Thoát hoàn toàn game.
2. Thay DLL tại `OptiScaler/custom/XeMFG XMX/dlssg/dlssg_to_xefg_intel_is_better.dll` và thư mục `xefg_mfg_programs` đi kèm bằng cùng một gói build mới. Không chỉ thay DLL hoặc chỉ thay manifest.
3. Cập nhật ASI đi kèm trong `OptiScaler/plugins` nếu đang dùng nó.
4. Giữ OptiScaler có bản vá chọn API RDR2 trước đó, giữ `libxess_fg.dll` đang dùng và cấu hình game.
5. Khởi động lại, bật FG và kiểm tra log: cần có `Vulkan XeFG context: ...`, không còn `missing SPIR-V translation`, `matched shifted SPIR-V row`, hoặc `init_-1000` lặp lại.

Log cũng có `VSync interval 1 not supported with FG`. Để tách vấn đề đồng bộ khỏi lỗi shader, lần thử đầu nên tắt VSync trong game/driver và tạm bỏ giới hạn FPS ngoài game. Không tự động thay đổi VSync hay giới hạn FPS của người dùng trong bản vá này. Chưa có bằng chứng đủ để khẳng định sửa shader sẽ loại bỏ mọi nguyên nhân khóa 25 FPS.

## Kiểm tra

Có kiểm tra hồi quy record đầu/cuối, null và hash hỏng; tra cứu hash trực tiếp/hai chiều/không tồn tại; kiểm tra payload thật và syntax/runtime trên Linux; workflow Windows build DLL và smoke-test exports. Các kiểm tra này không thay thế chạy game trên GPU. Chưa có phép đo FPS/GPU thực tế sau bản sửa.
