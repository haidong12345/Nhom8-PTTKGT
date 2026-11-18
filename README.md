# Nh-m-8-PTTKGT
"""
de06.py - Đề 6: Xếp lịch thi = bài toán tô màu đồ thị (min số màu, sắc số chi(G)).

Nội dung:
  * Bộ đọc DIMACS (.col) có kiểm tra: bỏ cạnh lặp / khuyên, báo số cạnh, kiểm tra chỉ số đỉnh.
  * Hàm kiểm tra lời giải: đếm cạnh xung đột + số màu đã dùng.
  * Cận dưới: clique tham lam (nhiều lần khởi tạo ngẫu nhiên)   omega(G) <= chi(G).
  * Heuristic xây dựng: Welsh-Powell, DSATUR.
  * Thuật toán chính xác: quay lui + cắt nhánh kiểu DSATUR, dùng cận dưới clique để dừng sớm.
  * Tìm kiếm cục bộ với k cố định (đổi màu 1 đỉnh xung đột; hoán đổi màu 2 đỉnh),
    đánh giá delta bằng bảng láng giềng gamma[v][c] (số láng giềng của v mang màu c).
  * TabuCol (Hertz & de Werra 1987): cập nhật bảng delta trong O(bậc) mỗi bước.
  * Quy trình giảm dần k: DSATUR -> TabuCol với k = best-1, best-2, ...
  * Thống kê nhiều lần chạy: best / trung bình / số lần đạt, thời gian.

Dùng:
  python de06.py vidu.col                       # chạy tất cả thuật toán trên 1 file
  python de06.py vidu.col --alg exact           # exact | wp | dsatur | tabu | ls | all
  python de06.py --selftest --time 10 --runs 5  # sinh myciel3-5, queen, gnp125 và chạy thử
  python de06.py g.col --out g.sol              # ghi lời giải (dòng 1: k; dòng 2: c(1..n))
"""