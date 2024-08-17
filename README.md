# 🌬️ 房间群通风网络反演项目

## 项目介绍

这是一个通过图注意力网络（Graph Attention Network, GAT）及变压器模型（Transformer）进行房间群通风网络反演的项目。该项目利用 GAT 处理空间序列数据，Transformer 处理时间序列数据，基于二氧化碳传感器实现实时的通风网络反演。

## 项目背景

在现代建筑中，通风系统的优化不仅能够提高居住者的舒适度，还能够显著节约能源。然而，传统的通风系统设计和优化方法通常依赖于复杂的物理模型和实验数据，难以实时应用。本项目旨在通过机器学习方法，特别是 GAT 和 Transformer 模型，来实现对房间群通风网络的实时反演，从而为通风系统的优化提供新的解决方案。

## 项目特点

- **图注意力网络（GAT）**：用于处理房间之间的空间序列数据，捕捉房间之间的复杂关系。
- **Transformer模型（Transformer）**：用于处理时间序列数据，捕捉通风系统随时间变化的动态特征。
- **实时性**：基于二氧化碳传感器数据，实现对通风网络的实时反演。

## 技术栈

- Python
- PyTorch
- 二氧化碳传感器数据

## 安装指南

1. 克隆本仓库到本地：
   ```bash
   git clone https://github.com/ITOTI-Y/Airflow-Reversal-Prediction.git
   ```
   
2. 进入项目目录：
   ```bash
   cd Airflow-Reversal-Prediction
   ```

3. 创建虚拟环境并激活：
   ```bash
   python -m venv venv
   source venv/bin/activate  # 对于 Windows 系统，使用 `venv\Scripts\activate`
   ```

4. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 配置二氧化碳传感器数据。
2. 结构化数据集
3. 输出结果

## 项目目录结构

```plaintext
Airflow-Reversal-Prediction
├── data                    #数据文件
├── errors                  #错误信息
├── src
│   ├── deepl               #模型文件
│   ├── utils               #通用工具
│   └── ventilation         #通风网络计算
└── tests                   #单元测试
    └── ventilation_test
```

## 贡献指南

欢迎对本项目提出意见和建议，您可以通过以下方式贡献：

1. 提交 Issue 报告问题或提出新功能建议。
2. 提交 Pull Request 贡献代码。

## 许可证

本项目使用 MIT 许可证，详情请参见 [LICENSE](./LICENSE) 文件。
