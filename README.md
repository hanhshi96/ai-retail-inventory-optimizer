# AI-Powered Retail Demand Forecasting & Inventory Reallocation

Đây là một prototype Applied AI kết hợp **machine learning demand forecasting** với logic **inventory reallocation** nhằm hỗ trợ ra quyết định tồn kho trong hệ thống bán lẻ nhiều cửa hàng.

## Business Problem

Trong hoạt động bán lẻ, mất cân bằng tồn kho thường xảy ra khi một số cửa hàng giữ quá nhiều hàng trong khi những cửa hàng khác lại thiếu hàng.

Project này thử nghiệm cách sử dụng Machine Learning và data-driven decision logic để:

- Dự báo nhu cầu bán hàng trong ngắn hạn
- Phát hiện nguy cơ thiếu hàng và tồn kho dư thừa
- Đề xuất điều chuyển hàng hóa giữa các cửa hàng
- Đo lường tác động tiềm năng của việc phân bổ lại tồn kho

Ý tưởng project xuất phát từ những bài toán phân bổ hàng hóa và tồn kho mình từng quan sát trong quá trình làm việc ở mảng e-commerce và business operations.

---

## Dataset

Project sử dụng **Corporación Favorita Store Sales dataset**.

Dataset bao gồm:

- Hơn **3 triệu daily sales observations**
- **54 stores**
- **33 product families**
- Thông tin promotion
- Store metadata bao gồm city và state

Daily sales được aggregate thành weekly demand ở cấp:

**Week × Store × Product Family**

> **Important:** Historical sales data là dữ liệu public thực tế.  
> Dataset không có actual inventory snapshots, vì vậy inventory levels trong giai đoạn reallocation được simulate để phục vụ mục đích minh họa.

---

## Project Architecture

```text
Real Retail Sales Data
        ↓
Data Cleaning & Weekly Aggregation
        ↓
Time-Series Feature Engineering
        ↓
Demand Forecasting
        ↓
Simulated Inventory Snapshot
        ↓
Shortage / Overstock Detection
        ↓
Geography-Aware Inventory Reallocation
        ↓
Business Impact Measurement
```

## 1. Demand Forecasting
Feature Engineering
Forecasting model sử dụng các feature:
- lag_1 — sales của 1 tuần trước
- lag_2 — sales của 2 tuần trước
- lag_4 — sales của 4 tuần trước
- lag_52 — sales tại thời điểm tương ứng khoảng 1 năm trước
- 4-week rolling demand average
- 8-week rolling demand average
- Promotion information
- Month
- Week of year
- Product-family encoding
Tuần cuối của dataset là một tuần chưa đầy đủ nên được loại bỏ trước khi modeling để tránh làm sai lệch model evaluation.
Project sử dụng time-based train/test split thay vì random split để tránh future information bị leakage vào quá trình training.

## 2. Models
Hai forecasting approaches được so sánh:
**Naive Baseline**
Giả định đơn giản: Next week's demand = Previous week's demand
**Machine Learning Model**
HistGradientBoostingRegressor
Model sử dụng historical demand, seasonality, promotion và calendar features để dự báo weekly demand.

## 3. Forecasting Results
Model	MAE	WAPE
Previous-week Baseline	314.99	9.49%
HistGradientBoosting	268.11	8.08%

Machine Learning model giúp giảm WAPE khoảng:
14.9% relative to the baseline

4. Inventory Simulation
Public sales dataset không có actual inventory snapshots.
Để xây dựng đầy đủ decision workflow, project tạo một lớp simulated inventory có khả năng reproduce.
Project sử dụng fixed random seed để đảm bảo mỗi store-product combination nhận cùng một simulated inventory level mỗi lần pipeline được chạy lại.
Target inventory được xác định theo công thức:
Target Inventory = Forecast Demand × 1.20
Phần 20% bổ sung được sử dụng như một simplified safety-stock buffer.

## 5. Inventory Classification
Mỗi store-product combination được phân loại thành:
SHORTAGE
Inventory thấp hơn 90% target.
BALANCED
Inventory nằm trong khoảng từ 90% đến 110% target.
OVERSTOCK
Inventory cao hơn 110% target.
Chỉ những store được xác định là rõ ràng OVERSTOCK mới được phép đóng vai trò donor trong quá trình reallocation.

## 6. Inventory Reallocation Engine
Hệ thống match những store có surplus inventory với những store đang thiếu cùng một product family.
Transfer priority:
1. Same city
2. Same state
3. Other states
Trong cùng một geographical priority, donor có surplus inventory lớn hơn sẽ được ưu tiên trước.
Geographical hierarchy này được sử dụng như một simplified proxy cho transportation cost.
Allocation engine hiện tại sử dụng heuristic logic và chưa phải là một global mathematical optimization model.

7. Business Impact
Prototype cuối cùng đạt được:
| Metric | Result |
|---|---:|
| Initial inventory shortage | 795,156 units |
| Available surplus | 832,154 units |
| Units reallocated | 708,508 units |
| Remaining shortage | 86,648 units |
| Shortage coverage | **89.10%** |
| Surplus utilization | **85.14%** |
| Recommended transfers | **963** |
**Allocation engine có thể cover khoảng 89.1% identified simulated inventory shortage thông qua store-to-store transfers.**

## 8. Repository Structure
ai-retail-inventory-optimizer/
│
├── assets/
│   ├── forecast_actual_vs_predicted.png
│   └── shortage_before_after.png
│
├── notebooks/
│   ├── 01_demand_forecasting.ipynb
│   └── 02_modular_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── forecasting.py
│   ├── inventory.py
│   ├── allocation.py
│   └── visualization.py
│
├── outputs/
│   ├── forecast_results.csv
│   ├── allocation_plan.csv
│   └── post_allocation_inventory.csv
│
├── README.md
├── requirements.txt
└── .gitignore

## 9. Notebook Design
01_demand_forecasting.ipynb
Notebook dạng step-by-step được sử dụng để học, thử nghiệm và hiểu toàn bộ workflow.
Notebook bao gồm:
- data preparation
- feature engineering
- forecasting
- evaluation
- inventory simulation
- reallocation logic
- business impact analysis
02_modular_pipeline.ipynb
Notebook dạng clean end-to-end pipeline, gọi reusable Python modules từ thư mục src/.
Notebook này thể hiện cách chuyển từ exploratory notebook sang một project structure dễ maintain hơn.

## 10. Code Architecture
Project tách responsibility theo từng module:
data.py
Phụ trách dataset loading, cleaning và weekly aggregation.
features.py
Phụ trách time-series feature engineering và chronological train/test splitting.
forecasting.py
Phụ trách baseline forecasting, Machine Learning training, prediction và evaluation metrics.
inventory.py
Phụ trách inventory simulation, target-stock calculation và inventory imbalance detection.
allocation.py
Phụ trách store-to-store transfer recommendations và allocation KPIs.
visualization.py
Phụ trách forecasting và inventory-impact visualizations.

## 11. Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Hugging Face Hub
- Git / GitHub

## 12. Reproducibility
Project sử dụng một số biện pháp để đảm bảo reproducibility:
- Fixed random seed cho inventory simulation
- Deterministic sorting trước khi inventory được generate
- Loại bỏ incomplete final sales week
- Chronological train/test split
- Allocation sanity check
Allocation pipeline kiểm tra điều kiện:
Initial shortage - Reallocated units = Remaining shortage

Final validation result:
795,156 - 708,508 = 86,648

## 13. Limitations
Đây là một Applied AI prototype, chưa phải production inventory optimization system.
Một số hạn chế hiện tại:
- Inventory data được simulate thay vì sử dụng dữ liệu tồn kho thực tế
- Dữ liệu được xử lý ở cấp product family thay vì individual SKU
- Geographical priority được sử dụng như proxy cho actual transport cost
- Allocation sử dụng heuristic rules thay vì global optimization
- Demand shocks và một số external factors chưa được mô hình hóa đầy đủ
- Logistics capacity và service-level constraints chưa được đưa vào hệ thống

## 14. Future Improvements
Các hướng phát triển tiếp theo có thể bao gồm:
- Sử dụng real SKU-level inventory data
- Bổ sung actual transportation distance và logistics cost
- Thêm holiday và external-demand features
- Thử nghiệm additional forecasting algorithms
- Áp dụng linear hoặc mixed-integer optimization
- Bổ sung service-level constraints
- Xây dựng interactive business dashboard

## Key Takeaway
Project này thể hiện một quy trình Applied AI for Business end-to-end:
Business Problem → Real Data → Machine Learning → Operational Decision → Measurable Business Impact
Mục tiêu không chỉ là xây dựng một forecasting model, mà còn là chuyển kết quả prediction thành actionable inventory decisions phục vụ bài toán business thực tế.
