# 💌 恋爱告急：告白成功率深度分析系统
> **Confession Analysis System (CAS) v2.0**

本项目是一个基于 **Python** 与 **Streamlit** 构建的情感量化工具。它结合了心理学经典理论 **斯滕伯格爱情三元论 (Sternberg's Triangular Theory of Love)** 与高斯分布数学模型，旨在通过数据化视角分析恋爱现状，并为用户预测最佳的“告白窗口期”。

---

## 🌟 核心功能

* **❤️ 爱情三元论量化分析**：通过“亲密 (Intimacy)”、“激情 (Passion)”、“承诺 (Commitment)”三个维度的加权算法，精准定位当前的爱情类型（如：完美之爱、浪漫之爱、虚幻之爱等）。
* **🎭 行为人格画像**：根据用户在决策时的犹豫程度与灵活性，自动识别用户属于“磨蹭型 (Hesitant)”、“骚动型 (Restless)”或“随缘型 (Spontaneous)”。
* **📈 告白成功率预测曲线**：利用高斯概率密度函数 $p(t) = A \cdot e^{-\frac{(t-t_0)^2}{2\sigma^2}}$，动态计算并展示成功率随时间变化的趋势。
* **☁️ 匿名数据同步**：集成 Google Sheets API，实时同步分析样本，支持情感调研数据收集。
* **📊 交互式可视化**：
    * **爱之三角雷达图**：直观展示关系的平衡度。
    * **时机分析曲线图**：标注理论巅峰时刻 ($T_{peak}$) 与用户实际预测行动时刻 ($T$) 的偏差。

---

## 🛠️ 技术栈

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Data Science**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
* **Visualization**: [Matplotlib](https://matplotlib.org/) (Agg Backend)
* **Database**: Google Sheets API (via `gspread`)

---

## 🚀 快速部署指南

### 1. 克隆仓库
```bash
git clone [https://github.com/your-username/love-analysis-system.git](https://github.com/your-username/love-analysis-system.git)
cd love-analysis-system
pip install streamlit numpy matplotlib gspread google-auth pandas
[gcp_service_account]
type = "service_account"
project_id = "你的项目ID"
private_key_id = "你的私钥ID"
private_key = "-----BEGIN PRIVATE KEY-----\n你的私钥内容\n-----END PRIVATE KEY-----\n"
client_email = "你的服务账号邮箱"
client_id = "..."
auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
client_x509_cert_url = "..."
