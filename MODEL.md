# 模型介绍 (Model Introduction)

## 系统使用的AI模型

本智能数据分析系统采用**大语言模型(LLM)驱动**的架构，核心NL2SQL（自然语言转SQL）能力由LLM提供。系统支持多种模型后端，通过OpenAI兼容API统一接入。

---

## 一、默认模型：DeepSeek

| 属性 | 说明 |
|------|------|
| **模型名称** | `deepseek-chat` (DeepSeek-V3) |
| **提供商** | DeepSeek (深度求索) |
| **API地址** | `https://api.deepseek.com/v1` |
| **模型类型** | MoE (混合专家) 大语言模型 |
| **上下文窗口** | 64K tokens |
| **定价** | ￥1/百万tokens (输入) / ￥2/百万tokens (输出) |
| **中文能力** | 优秀 —— 原生中文训练，SQL生成质量高 |

### 为什么选择DeepSeek

1. **性价比极高**：API价格仅为GPT-4o的1/10，适合批量评测和持续使用
2. **中文友好**：对中文自然语言问题的理解准确，SQL生成质量好
3. **OpenAI兼容**：API与OpenAI格式完全兼容，零代码切换
4. **推理速度快**：MoE架构，生成速度可达50+ tokens/s

---

## 二、可选模型

系统通过统一LLM抽象层（`backend/llm/client.py`）支持以下替代模型：

### 2.1 GPT-4o / GPT-4o-mini (OpenAI)

```bash
setx OPENAI_API_KEY "sk-your-openai-key"
setx LLM_MODEL "gpt-4o"
setx OPENAI_BASE_URL "https://api.openai.com/v1"
```

| 优势 | 劣势 |
|------|------|
| 综合能力最强 | 价格较高 |
| SQL生成稳定 | 国内访问需代理 |

### 2.2 九天大模型 (中国移动)

九天平台「应用接入」新建应用并关联推理服务后，复制 **AppCode**，通过 `Authorization: Bearer <AppCode>` 鉴权调用 `generate_stream` 推理接口。

```bash
# 配置 AppCode (九天平台「应用接入」复制)
setx JIUTIAN_APP_CODE "your-appcode"
```

默认 provider 已切到 jiutian，见 `config/settings.yaml`：

```yaml
llm:
  provider: "jiutian"
  model: ""                              # 单模型部署可不填
  base_url: "http://127.0.0.1:8090/generate_stream"
```

| 优势 | 劣势 |
|------|------|
| 运营商自有，内网可用 | 模型能力可能不及通用大模型 |
| 数据不出内网，安全性高 | 需本地推理服务在线 (8090) |

### 2.3 Qwen (通义千问)

```bash
setx DEEPSEEK_API_KEY "your-qwen-key"
setx LLM_MODEL "qwen-turbo"
setx OPENAI_BASE_URL "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

| 优势 | 劣势 |
|------|------|
| 阿里云生态，中文优秀 | API格式略有差异 |
| 免费额度充足 | — |

### 2.4 本地模型 (Ollama / vLLM)

```bash
setx LLM_MODEL "qwen2.5:7b"
setx OPENAI_BASE_URL "http://localhost:11434/v1"
setx DEEPSEEK_API_KEY "ollama"   # Ollama不需要真实key
```

---

## 三、模型在系统中的角色

系统的一条完整链路中，LLM在以下环节被调用：

```
用户提问
  │
  ├── [LLM调用1] 意图分类 (intent_classifier.py)
  │   将问题归类为: SQL查询/异常检测/预测/报告/闲聊
  │   输入: ~50 tokens | 输出: ~5 tokens
  │
  ├── [LLM调用2] NL2SQL生成 (nl2sql_generator.py)
  │   根据Schema+问题+意图生成SQL语句
  │   输入: ~500-2000 tokens | 输出: ~50-200 tokens
  │   这是最关键的LLM调用，直接影响SQL质量
  │
  ├── [LLM调用3] 结果解释 (result_explainer.py)
  │   将查询结果翻译为自然语言说明
  │   输入: ~300 tokens | 输出: ~100-300 tokens
  │
  └── [LLM调用4] 报告生成 (report_generator.py, 可选)
      根据问题+数据生成分析报告(标题/摘要/洞察/建议)
      输入: ~1000-3000 tokens | 输出: ~500-1000 tokens
```

### 各环节对模型能力的要求

| 环节 | 关键能力 | 推荐模型 |
|------|----------|----------|
| 意图分类 | 文本分类、理解力 | DeepSeek-Chat (够用) |
| NL2SQL生成 | **SQL生成、Schema理解、逻辑推理** | DeepSeek-V3 / GPT-4o |
| 结果解释 | 数据理解、自然语言生成 | DeepSeek-Chat (够用) |
| 报告生成 | 综合分析、洞察提炼、结构化输出 | GPT-4o (最优) / DeepSeek-V3 |

---

## 四、Prompt工程设计

### 4.1 NL2SQL System Prompt 核心策略

```
你是一个SQL查询生成专家。
- 根据用户问题和表结构生成精确的SQL
- 仅生成SELECT语句，禁止修改数据
- 使用中文别名使结果可读
- 数值计算注意类型转换
- 优先使用标准SQL语法
```

### 4.2 意图专用Prompt

系统为不同意图设计了专用System Prompt：

| 意图 | Prompt特点 |
|------|------------|
| `SQL_QUERY` | 标准SQL生成，注重精确性 |
| `ANOMALY_DETECT` | 引导生成全量数值查询，便于后续统计检测 |
| `FORECAST` | 引导生成历史时间序列数据查询 |
| `REPORT` | 引导生成聚合汇总查询 |

### 4.3 Schema注入格式

```
## 表: employees
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 员工ID(主键) |
| name | TEXT | 姓名 |
| salary | REAL | 薪资 |
| dept_id | INTEGER | 部门ID(外键→departments.id) |

## 已知表关联:
- employees.dept_id ↔ departments.id (外键约束)
```

---

## 五、模型切换指南

### 切换步骤

1. **设置环境变量**（选择一种方式）：

```bash
# Windows CMD - 切换到GPT-4o
setx DEEPSEEK_API_KEY "sk-your-openai-key"
setx LLM_MODEL "gpt-4o"
setx OPENAI_BASE_URL "https://api.openai.com/v1"
```

```bash
# Windows PowerShell
[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'sk-xxx', 'User')
[Environment]::SetEnvironmentVariable('LLM_MODEL', 'gpt-4o', 'User')
```

2. **重启终端和后端服务**

3. **验证**：
```bash
curl http://localhost:8000/api/health
# 确认 api_key_configured: true
```

### 环境变量完整列表

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `JIUTIAN_APP_CODE` | 是(默认jiutian) | — | 九天平台 AppCode (Bearer 鉴权) |
| `DEEPSEEK_API_KEY` | 否(切deepseek时) | — | DeepSeek API密钥 |
| `LLM_MODEL` | 否 | `deepseek-chat` | 模型名称 |
| `OPENAI_BASE_URL` | 否 | `https://api.deepseek.com/v1` | API地址 |
| `LLM_TEMPERATURE` | 否 | `0.1` | 生成温度(0-1) |
| `LLM_MAX_TOKENS` | 否 | `2000` | 最大输出tokens |

---

## 六、性能参考

| 模型 | 平均延迟 | SQL准确率(金融) | SQL准确率(医疗) | 成本/千次查询 |
|------|----------|-----------------|-----------------|---------------|
| DeepSeek-V3 | ~2-4s | ~82% | ~78% | 约￥3-5 |
| GPT-4o | ~3-6s | ~88% | ~85% | 约￥30-50 |
| GPT-4o-mini | ~1-3s | ~80% | ~76% | 约￥2-3 |
| Qwen-Turbo | ~2-4s | ~80% | ~77% | 约￥1-3 |

> *准确率为内部测试参考值，实际表现因数据结构和问题复杂度而异*

---

## 七、安全说明

- **API Key绝不存储在代码文件中**：密钥仅保存在系统环境变量（Windows）或shell配置文件（Linux/Mac）
- **密钥验证在启动时进行**：`backend/config.py` 的 `check_api_key()` 检查密钥是否为占位符
- **所有LLM调用均为只读**：系统不向模型发送任何写入数据库的指令
- **SQL校验层拦截危险操作**：即使模型生成DROP/DELETE，也会被校验器拦截
