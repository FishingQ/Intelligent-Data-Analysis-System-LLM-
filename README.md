# 智能数据分析系统

> 中国移动"九天·梧桐"AI+数据创新赛道 —— 赛题1  
> AI驱动的对话式数据分析工具 | 自然语言 → SQL → 可视化 → 报告

## 功能概览

| 能力 | 等级 | 状态 |
|------|------|------|
| 自然语言→SQL查询 | 初级(必选) | ✅ 完成 |
| 多数据源适配(SQLite/Excel/CSV/MySQL/PG) | 初级(必选) | ✅ 完成 |
| 统计分析 + 关联查询 + 公式计算 | 中级(可选) | ✅ 完成 |
| 智能可视化(折线/柱状/饼图/散点) | 中级(可选) | ✅ 完成 |
| 决策报告自动生成 | 高级(可选) | ✅ 完成 |
| 异常检测 + 时序预测 + 数据挖掘 | 高级(可选) | ✅ 完成 |
| 多轮对话 + 历史管理 | 高级(可选) | ✅ 完成 |

## 项目结构

```
智能数据分析系统/
├── backend/
│   ├── api/              # FastAPI 路由 (chat/datasources/conversations)
│   │   ├── main.py       # 应用入口 + 生命周期管理
│   │   └── routes/       # chat.py, datasources.py, conversations.py
│   ├── llm/              # LLM调用封装 (LangChain)
│   │   ├── client.py     # ChatOpenAI 统一客户端
│   ├── nlp/              # NL2SQL + 意图识别
│   │   ├── intent_classifier.py    # 查询意图分类
│   │   ├── schema_mapper.py        # Schema映射与JOIN建议
│   │   ├── nl2sql_generator.py     # NL→SQL生成器
│   │   ├── sql_validator.py        # SQL安全校验
│   │   ├── result_explainer.py     # 结果自然语言解释
│   │   └── conversation_manager.py # 多轮对话管理
│   ├── data_sources/     # 多数据源适配器
│   │   ├── factory.py    # 适配器工厂
│   │   ├── base.py       # 基类接口
│   │   ├── sqlite.py     # SQLite适配器
│   │   ├── duckdb_excel.py  # Excel/CSV (DuckDB引擎)
│   │   └── sqlalchemy_mysql.py  # MySQL/PostgreSQL
│   ├── analyzer/         # 分析计算引擎
│   │   ├── stats_calculator.py   # 统计计算
│   │   ├── query_executor.py     # 查询执行器
│   │   ├── join_analyzer.py      # JOIN关联发现
│   │   ├── formula_engine.py     # 公式引擎
│   │   ├── anomaly_detector.py   # 异常检测 (IQR/Z-Score/IsolationForest)
│   │   ├── forecaster.py         # 时序预测 (Prophet/MA)
│   │   └── report_generator.py   # LLM报告生成
│   ├── visualizer/       # 图表 + 报告
│   │   ├── chart_recommender.py  # 图表智能推荐
│   │   └── echarts_builder.py    # ECharts配置构建
│   ├── training/         # 训练数据管理
│   │   ├── __init__.py
│   │   └── dataset_loader.py    # 结构化+非结构化数据集加载
│   └── shared/           # 统一数据模型
│       └── schemas.py    # Pydantic模型定义
├── frontend/
│   └── app.py            # Streamlit Web界面 (对话+图表+报告)
├── scripts/
│   ├── import_datasets.py  # 训练数据导入
│   └── run_eval.py         # 批量评测脚本
├── config/
│   └── settings.yaml     # 系统配置
├── data/                 # 数据文件目录
├── docs/                 # 文档
├── tests/                # 测试
├── start.bat             # Windows一键启动
├── start.sh              # Linux/Mac一键启动
├── Dockerfile            # 后端Docker镜像
├── Dockerfile.frontend   # 前端Docker镜像
├── docker-compose.yml    # Docker Compose编排
└── requirements.txt      # Python依赖
```

## 快速启动

### 方式一：启动脚本（推荐）

**Windows:**
```bash
双击 start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh && ./start.sh
```


### 方式二：Docker部署

```bash
# 仅核心服务（后端+前端）
docker-compose up -d

# 包含MySQL/PostgreSQL
docker-compose --profile with-db up -d
```

### 方式三：手动启动

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入九天平台 AppCode（在「应用接入」复制）
# Windows: setx JIUTIAN_APP_CODE "你的AppCode"
# Linux/Mac: export JIUTIAN_APP_CODE="你的AppCode"
```

#### 3. 启动后端

```bash
uvicorn backend.api.main:app --reload --port 8000
# API文档: http://localhost:8000/api/docs
```

#### 4. 启动前端

```bash
streamlit run frontend/app.py
# 界面: http://localhost:8501
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **后端框架** | Python FastAPI | 高性能异步API |
| **前端** | Streamlit + ECharts | 对话式Web界面 |
| **LLM** | 九天大模型 / DeepSeek / OpenAI | 多后端（九天 generate_stream / OpenAI 兼容） |
| **数据处理** | Pandas + NumPy + DuckDB | 多格式数据分析引擎 |
| **数据库** | SQLite / MySQL / PostgreSQL | 关系型数据源 |
| **机器学习** | Scikit-learn + Prophet | 异常检测 + 时序预测 |
| **容器化** | Docker + Docker Compose | 一键部署 |

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 核心对话接口（NL→SQL→执行→可视化） |
| `/api/datasources/upload` | POST | 上传数据文件 |
| `/api/datasources` | GET | 列出已注册数据源 |
| `/api/datasources/{id}` | DELETE | 删除数据源 |
| `/api/conversations` | GET | 列出所有会话 |
| `/api/conversations/{id}` | GET | 获取会话详情 |
| `/api/conversations/{id}` | DELETE | 删除会话 |
| `/api/health` | GET | 健康检查 |
| `/api/info` | GET | 系统信息 |

## 开发阶段

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 项目骨架 + 基础设施 | ✅ 完成 |
| 1 | NL2SQL核心链路 | ✅ 完成 |
| 2 | 可视化 + 多数据源 + 统计分析 | ✅ 完成 |
| 3 | 异常检测 + 预测 + 报告 + JOIN + 公式 | ✅ 完成 |
| 4 | 测试 + 文档 + 打包 + 部署 | ✅ 完成 |

## 评测

```bash
# 评测单个数据集
python scripts/run_eval.py --dataset 金融 --limit 10

# 评测全部
python scripts/run_eval.py --all --limit 5

# 输出报告
python scripts/run_eval.py --all --output eval_report.json
```

## License

MIT
