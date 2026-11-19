# Nhóm 8 – PTTKGT – Đề 6: Xếp lịch thi (bài toán tô màu đồ thị)

Mỗi môn thi là một đỉnh; hai môn có sinh viên chung được nối cạnh. Tô màu các đỉnh (màu = ca thi) sao cho hai đỉnh kề khác màu và dùng ít màu nhất (sắc số χ(G)).

## Cấu trúc thư mục

```
de06.py            chương trình chính (dòng lệnh) + máy chủ cho trang web
visualize/         trang web trực quan hoá (index.html, chạy trên trình duyệt)
data/              đồ thị mẫu định dạng DIMACS (.col)
vidu.sol           lời giải mẫu cho ví dụ nhỏ của đề
report/            báo cáo
```

## Thuật toán (de06.py)

- Đọc DIMACS có kiểm tra, kiểm chứng lời giải (đếm cạnh xung đột, số màu).
- Cận dưới clique ω(G) ≤ χ(G).
- Welsh–Powell, DSATUR.
- Quay lui cắt nhánh (chính xác) với cận dưới clique.
- Tìm kiếm cục bộ với k cố định (đổi màu 1 đỉnh, hoán đổi màu 2 đỉnh), bảng delta γ[v][c].
- TabuCol (Hertz – de Werra 1987), quy trình giảm dần k.
- GA lai HEA: toán tử GPX (Galinier – Hao 1999) + TabuCol.
- Bộ lịch thi Toronto (.crs/.stu): tối thiểu số ca thi, sau đó giảm chi phí khoảng cách (phạt 2^(5−d)).

## Cách chạy

Yêu cầu: Python 3 (không cần thư viện ngoài).

```
python de06.py data/vidu.col --out vidu.sol
python de06.py data/myciel4.col --alg exact,tabu,hea --time 30 --runs 10
python de06.py --selftest --time 10 --runs 5
python de06.py --toronto data/tor_demo --time 60 --slots 50
```

`--alg` nhận: `wp`, `dsatur`, `exact`, `tabu`, `ls`, `hea`, `all`.
File lời giải: dòng 1 là số màu k, dòng 2 là c(1), …, c(n).

## Trang web trực quan hoá

```
python de06.py --web
```

Lệnh này mở `http://127.0.0.1:5500/visualize/index.html`. Trang có: đồ thị đổi màu theo thời gian thực, đường hội tụ số màu, bảng đối chiếu thuật toán và validator. Danh sách đồ thị có sẵn các file `.col` trong `data/`. Cũng có thể mở file `.col` bất kỳ bằng nút chọn file.

Có thể bấm đúp `visualize/index.html` để mở trực tiếp (khi đó chỉ dùng được các đồ thị mẫu sinh sẵn và chọn file `.col` thủ công, không tự đọc được `data/`).
