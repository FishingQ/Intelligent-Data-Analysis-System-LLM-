"""
FastAPI 主入口
启动: uvicorn backend.api.main:app --reload --port 8000
"""
import sys
import os
import logging

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- 创建应用 ----
app = FastAPI(
    title="智能数据分析系统 API",
    description="AI驱动的对话式数据分析工具 —— 赛题1",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 初始化服务
# ============================================================

def _init_services():
    """从系统环境变量加载配置，初始化 LLM 客户端和所有服务模块"""
    from backend.config import get_llm_config, check_api_key
    from backend.llm.client import create_llm_client

    # 加载 LLM 配置（从系统环境变量/注册表读取密钥）
    llm_cfg = get_llm_config()

    # 安全提示
    if not check_api_key():
        logger.warning("=" * 60)
        logger.warning("  API Key 未配置!")
        logger.warning("  请通过系统环境变量设置您的 LLM API Key:")
        logger.warning("")
        logger.warning("    Windows CMD:")
        logger.warning('      setx DEEPSEEK_API_KEY "sk-your-key"')
        logger.warning("")
        logger.warning("    Windows PowerShell:")
        logger.warning("      [Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'sk-xxx', 'User')")
        logger.warning("")
        logger.warning("  设置后需重启终端和本服务方可生效。")
        logger.warning("=" * 60)

    llm_client = create_llm_client(
        provider=llm_cfg["provider"],
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        timeout=llm_cfg["timeout"],
        max_retries=llm_cfg["max_retries"],
    )
    cred = llm_cfg.get("api_key") or ""
    logger.info(
        f"LLM客户端(LangChain): provider={llm_cfg['provider']}, model={llm_cfg['model']}, "
        f"api_key={'***' + cred[-4:] if cred and len(cred) > 8 else '(未配置)'}"
    )

    # 初始化 chat 路由所需的服务
    from backend.api.routes.chat import init_chat_services
    init_chat_services(llm_client)

    # 初始化知识库 RAG 服务（共享同一 LLM 客户端）
    from backend.api.routes.rag import init_rag_services
    init_rag_services(llm_client)

    return llm_client


# 应用启动
@app.on_event("startup")
async def startup():
    """应用启动事件"""
    logger.info("🚀 智能数据分析系统 API 启动中...")
    try:
        _init_services()
        logger.info("✅ 所有服务模块初始化完成")
    except Exception as e:
        logger.warning(f"⚠️ 部分服务初始化失败: {e}")
        logger.warning("系统将以降级模式运行——NL2SQL功能不可用")


# ---- 注册路由 ----
from backend.api.routes import chat, datasources, conversations, rag

app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(datasources.router, prefix="/api", tags=["数据源"])
app.include_router(conversations.router, prefix="/api", tags=["会话历史"])
app.include_router(rag.router, prefix="/api", tags=["知识库"])

logger.info("路由已注册: /api/chat, /api/datasources/*, /api/conversations/*, /api/rag/*")


# ---- 健康检查 ----
@app.get("/api/health")
async def health_check():
    """健康检查"""
    from backend.config import check_api_key
    return {
        "status": "ok",
        "version": "1.0.0",
        "phase": "4 - 完整交付",
        "api_key_configured": check_api_key(),
    }


@app.get("/api/info")
async def system_info():
    """系统信息"""
    return {
        "name": "智能数据分析系统",
        "description": "AI驱动的对话式数据分析工具",
        "version": "1.0.0",
        "phase": "4 - 完整交付 (NL→SQL→执行→解释→图表→异常检测→预测→报告→对话管理)",
        "security": {
            "api_key_storage": "系统环境变量 (不在任何文件中存储密钥)",
            "providers": ["deepseek", "jiutian", "openai", "qwen"],
        },
        "capabilities": {
            "data_sources": ["sqlite", "excel", "csv", "mysql", "postgresql"],
            "nl2sql": True,
            "intent_classification": True,
            "result_explanation": True,
            "visualization": "full",
            "multi_table_join": True,
            "anomaly_detection": True,
            "forecast": True,
            "report_generation": True,
            "formula_engine": True,
            "conversation_manager": True,
            "conversation_history_api": True,
        },
    }


# ---- 启动入口 ----
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 启动智能数据分析系统 API 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
