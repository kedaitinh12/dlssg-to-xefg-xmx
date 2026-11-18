# Bản vá RDR2: chọn đúng API cho Streamline

Bản vá này sửa **OptiScaler**, không thay kernel CUDA/XMX/DP4a của mod XeFG.
Nguồn gốc: OptiScaler commit `03b37db73979e042bce4be59f44d37f348ac6905`, đúng phiên bản trong log được cung cấp.
Đây là bản tùy chỉnh để kiểm tra lỗi; chưa được xác minh trong game trên GPU thật.

## Bằng chứng từ log

- 18:54:29.622: RDR2.exe gọi D3D12CreateDevice trước khi bắt đầu Vulkan.
- 18:54:29.790: OptiScaler khởi tạo Streamline trên đường D3D12.
- 18:54:34.260: `Streamline is already initialized for D3D12`, tiếp theo `Can't init Streamline, DLSSG won't be available`.
- Log XeFG cho thấy đủ 83 chương trình Vulkan, device NVIDIA được chuẩn bị và runtime báo ready.
- Không có dòng `Vulkan: feature created` trong log XeFG. Các lệnh EvaluateFeature handle 1000000 trong OptiScaler thuộc DLSS upscaler, không chứng minh frame generation đã chạy.

Vì Streamline không hỗ trợ khởi tạo lại an toàn sang API khác trong tiến trình này, bản vá tránh chọn D3D12 từ device thăm dò. Không gọi shutdown/reinit hoặc ép đổi API sau khi Streamline đã chạy.

## Thay đổi

- Thêm quirk chỉ cho `rdr2.exe` và `playrdr2.exe`.
- Hoãn `InitWithD3D12` ở hook device. Khi chơi RDR2 bằng D3D12, constructor DLSSG_Dx12 tại đường swapchain vẫn khởi tạo Streamline.
- Vẫn nạp sớm custom DLSSG provider ngay từ device thăm dò, để hook XeFG thấy cả vkCreateInstance và vkCreateDevice. Chỉ hoãn slInit, không hoãn hook của mod.
- Chế độ Vulkan dùng InitWithVulkan hiện có trước khi tạo device. Các game khác giữ đường khởi tạo cũ.

## Build trên repository của mod

Commit các file trong gói vào nhánh đang dùng, rồi chạy workflow **Build OptiScaler RDR2 Vulkan fix**.
Workflow checkout đúng commit OptiScaler nêu trên, tải submodule, kiểm tra patch áp dụng sạch, build Windows x64 Release và xuất artifact **OptiScaler-RDR2-Vulkan-fix**.
Không cần build lại DLL XeFG để sửa điểm chặn này. Workflow không ghi đè cấu hình game, không phát hành release.

## Cài để kiểm tra

1. Thoát hẳn RDR2 và launcher đang giữ DLL; lưu bản sao DLL OptiScaler đang dùng.
2. Log cho thấy OptiScaler đang chạy dưới tên `winmm.dll`. Đổi tên **OptiScaler.dll trong artifact mới** thành `winmm.dll`, thay đúng file OptiScaler cũ cạnh RDR2.exe. Không thay DLL trong Windows/System32.
3. Giữ thư mục OptiScaler/streamline, custom/XeMFG XMX, shader payload và libxess_fg.dll đang dùng. Không chép OptiScaler.dll vào thư mục custom/XeMFG XMX/dlssg.
4. Giữ FGInput=Upscaler, FGOutput=DLSSG, FGNvngxReplacement=XeMFG XMX, LoadAsiPlugins=true và VulkanHooks=true cho plugin. Không thay toàn bộ OptiScaler.ini bằng cấu hình mặc định.
5. Khởi động lại game với Vulkan, bật frame generation và vào gameplay để kiểm tra. Việc đổi menu FG mà không restart không thể sửa Streamline đã chọn sai API trong tiến trình cũ.

Trong log mới phải có `Deferring Streamline API selection for RDR2` và `Preloading custom DLSSG`, không còn lỗi `already initialized for D3D12`. Sau khi bật FG cần kiểm tra thêm `Vulkan: feature created` trong log XeFG; runtime ready đơn thuần chưa đủ.

Nếu FG vẫn không chạy, gửi lại cả hai log của lần chạy mới. Bản vá xử lý điểm chặn đã chứng minh trong log; việc execute shader, image layout, HDR và tốc độ trên RTX 3070 cần kiểm tra thực tế sau khi tạo được feature.

Để hoàn tác, thoát game và khôi phục winmm.dll đã sao lưu. Không cần đổi model hoặc shader.
