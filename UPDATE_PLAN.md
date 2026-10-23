# CR-sqlmap — Kế hoạch cập nhật (giai đoạn thử nghiệm + code), v1

*Dựa trên 3 file nghiên cứu: `vuot_libinjection_tong_hop.md`, `mutation_op_catalog_libinjection.md`, `related_work.md`. Mục tiêu: nhắm **libinjection** (CRS 3.3.2 rule 942100) làm target, giữ Ouroboros **faithful**, chỉ bổ sung "lớp libinjection". Cập nhật 2026-10-23.*

## Nguyên tắc xuyên suốt
- **Faithful Ouroboros:** không sửa math spec đã khoá (`CR-sqlmap_Ouroboros_math_spec.md`). Mọi thứ mới (oracle, R_LIB, fitness fingerprint) là **lớp mở rộng khai báo rõ ràng**, bật/tắt bằng cờ, mặc định không phá đường chạy cũ.
- **Hộp đen vẫn là hộp đen:** agent chỉ thấy 200/403 từ WAF. libinjection offline chỉ là **oracle shaping reward/fitness**, không phải tín hiệu WAF.
- **2 oracle cục bộ:** (a) libinjection offline → fingerprint rời-bảng hay chưa; (b) DVWA không-WAF `127.0.0.1:8081` → payload có còn tiêm được không. "Ứng viên vàng" = đạt cả hai.

---

## Phase 1 — Dựng oracle libinjection offline  *(nền tảng, làm trước)*
**Vì sao trước:** mở khoá reward dày, fitness GA, và validate action-space trước khi đốt vòng train.
1. Xác định **đúng version libinjection** nhúng trong image `owasp/modsecurity-crs:nginx` đang chạy (đọc hash/commit trong container), rồi clone đúng version đó → bảng fingerprint khớp rule 942100.
2. Build trên Kali: Python binding (`import libinjection`) + `fptool`/`sqli_cli` (xem token + fingerprint + verdict).
3. **File mới `libinj_oracle.py`:** hàm `verdict(payload) -> {sqli: bool, fingerprint: str, tokens: [...]}`. Có cache.
4. **`relay.py`:** thêm endpoint cục bộ `/libinj` (chạy oracle, KHÔNG qua WAF) để agent trên Kaggle gọi lấy tín hiệu dày mà không tốn vòng WAF.
- **Kiểm chứng Phase 1:** chạy lại các payload đã biết (`1' #` lọt; `1' union select…#` bị chặn) qua oracle, xác nhận verdict khớp hành vi WAF thực đã đo.

## Phase 2 — Thiết kế lại action space + viết tamper còn thiếu
**File:** `kaggle_waf_rl.py` (OPERATORS), `executor.py` (tamper pipeline).
1. Thay 7 preset tamper generic hiện tại bằng **preset ưu tiên libinjection** từ catalog §C/§D:
   - *Ưu tiên 1 (phá tokenizer):* versioned-comment, scientific-notation, function-substitution (hàm lạ), backtick+subquery, special-char tautology, control-char whitespace, quotation-mark encoding.
   - *Ưu tiên 2 (né CRS regex):* comment-injection, clause-concat, schema-split, hex, char/unicode encoding.
   - *Hạ ưu tiên:* case-swap (vô dụng với libinjection), comment-rewrite đứng riêng.
2. **Viết tamper mới** cho các op sqlmap chưa có (map ↔ catalog): `#2 scientific-notation`, `#3 ODBC curly-brace`, `#4 special-char tautology`, `#21 backtick-subquery`, `#15 redundant-keyword-omission`. Đặt trong thư mục tamper riêng, nạp qua `apply_tamper_chain()` (đã hỗ trợ chain).
3. Giữ cấu trúc Discrete; cập nhật `N_ACTIONS` + `decode_action` + assert cho khớp số preset mới.
- **Kiểm chứng Phase 2:** mỗi preset chạy qua oracle libinjection + gửi `:8081` → log preset nào đẩy `sqli=False` mà vẫn tiêm được.

## Phase 3 — Reward dày dùng libinjection làm oracle  *(sửa reward≈0)*
**File:** `kaggle_waf_rl.py` (`RewardCombiner`), `executor.py` (probe trả thêm trường fingerprint).
1. Thêm thành phần dày **`R_LIB`** (khai báo là lớp mở rộng libinjection):
   `R_LIB = w·1[fingerprint rời badset] + …` theo công thức §D: `r = w1·(fp_out) + w2·(WAF 200) + w3·(sqlmap xác nhận tiêm) − w4·len`.
2. Gắn `R_LIB` vào bó **surrogate dày** (cùng chỗ R_RUL/R_REG), có cờ bật/tắt và trọng số môi-trường; **không đổi** công thức R_SYS/R_tot/anneal β đã khoá. Khi tắt cờ → reward về đúng Ouroboros gốc.
- **Kiểm chứng Phase 3:** chạy ngắn trên Kali, xác nhận reward có gradient (khác ≈0) ngay cả khi chưa lọt WAF.

## Phase 4 — GA faithful Ouroboros + warm-start có định hướng
**File mới:** `ga.py`. Cập nhật `warmstart.py`.
1. GA đúng spec: endpoint-erosion → TF-IDF → DBSCAN → 16-gene hex → (value,length) IR → LCS → regex; fitness `L(r)=α·FN+β·FP+γ_len·len+γ_c·Cplx` (giữ nguyên hệ số đã khoá).
2. **Biến thể guided (khai báo rõ):** seed quần thể từ **payload sqlmap đã xác nhận bypass** + lọc bằng oracle libinjection; bổ sung "fingerprint-distance" vào fitness như **lớp mở rộng** (ghi rõ Ouroboros gốc giả định WAF regex — §2.4/§2.6 related_work).
3. `warmstart.py`: behavior-cloning từ quỹ đạo tamper thành công (found=True qua WAF).
- **Kiểm chứng Phase 4:** GA sinh ≥1 regex/biến thể mới, feed FA2RNN (`fa2rnn.py`) ra ψ hợp lệ.

## Phase 5 — Tích hợp + chạy thử nhỏ
1. Chạy end-to-end trên Kali (N_ENVS nhỏ, vài trăm step) xác nhận: oracle nối, action-space mới, R_LIB có gradient, GA seed vào BC.
2. Quan sát found-rate qua WAF so với baseline cũ (block≈0.972, found=False).

## Phase 6 — Cập nhật tài liệu
- Addendum vào math spec: mục "Libinjection extension" (oracle, R_LIB, fingerprint-distance) — **không sửa phần đã khoá**.
- `OVERVIEW.md`, `README.md`: action-space mới + sơ đồ 2-oracle.
- Nhúng `related_work.md` vào paper.

---

## File thay đổi (tóm tắt)
| File | Loại | Nội dung |
|---|---|---|
| `libinj_oracle.py` | MỚI | oracle libinjection offline (verdict+fingerprint+token) |
| `relay.py` | sửa | endpoint `/libinj` cục bộ |
| `executor.py` | sửa | 5 tamper mới + probe trả fingerprint |
| `kaggle_waf_rl.py` | sửa | OPERATORS mới, N_ACTIONS, R_LIB trong RewardCombiner |
| `ga.py` | MỚI | GA faithful + seed guided + fitness fingerprint |
| `warmstart.py` | sửa | BC từ quỹ đạo tamper thành công |
| `fa2rnn.py` | (nhẹ) | nhận regex từ GA |
| math spec / OVERVIEW / README | doc | addendum lớp libinjection |

## Hạ tầng
- Push cloud→GitHub đang bị chặn (403) → **Minh tự `git push`** từ PC. File ghi thẳng vào `C:\Users\Admin\Documents\Claude\cr-sqlmap` qua phiên Remote Control.
- Oracle + build libinjection chạy trên Kali (SSH). DVWA-direct `:8081` đã dựng sẵn để verify tiêm.

## Thứ tự đề xuất
**1 → 2 → 3 → (chạy thử) → 4 → 5 → 6.** Phase 1–3 gỡ đúng nút thắt reward≈0; Phase 4 là phần Ouroboros nặng.
