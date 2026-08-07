"""
配置加载器
安全地从系统环境变量加载配置 —— 不在任何文件中存储密钥

优先级: 系统环境变量 > .env文件(仅本地开发)
"""
import os
import sys
import yaml
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 密钥管理 —— 只从系统环境变量读取，绝不写入文件
# ============================================================

# 敏感配置项 → 对应的环境变量名
# 按优先级顺序排列（第一个匹配到的生效）
_SECRET_ENV_MAP = {
    "llm.api_key": [
        "DEEPSEEK_API_KEY",
        "Deepseek_apikey",
        "deepseek_api_key",
        "DEEPSEEK_KEY",
    ],
    "llm.jiutian_api_key":  ["JIUTIAN_API_KEY"],
    "llm.openai_api_key":   ["OPENAI_API_KEY"],
    "llm.qwen_api_key":     ["QWEN_API_KEY"],
}

# 非敏感默认值
_DEFAULTS = {
    "llm.provider":         "deepseek",
    "llm.model":            "deepseek-chat",
    "llm.base_url":         "https://api.deepseek.com/v1",
    "llm.temperature":      0.1,
    "llm.max_tokens":       4096,
    "llm.timeout":          60,
    "llm.max_retries":      2,
    "server.host":          "0.0.0.0",
    "server.port":          8000,
    "data.upload_dir":      "./data/uploads",
    "data.max_file_size_mb": 50,
    "analysis.max_sql_length":     10000,
    "analysis.max_result_rows":    10000,
    "analysis.query_timeout_seconds": 30,
    "analysis.max_conversation_turns": 20,
}


def _load_env_file() -> Dict[str, str]:
    """
    尝试加载 .env 文件（仅本地开发辅助，不覆盖系统环境变量）

    返回 .env 中但系统环境变量中不存在的补充值
    """
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_file):
        return {}

    extras = {}
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # 只补充系统环境变量中不存在的
                if key and key not in os.environ:
                    extras[key] = val
    except Exception as e:
        logger.debug(f"读取 .env 文件失败（可忽略）: {e}")

    return extras


def _read_from_registry(key_name: str) -> str:
    """Windows: 从注册表读取用户环境变量 (setx 写入但尚未生效的)"""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    if name.upper() == key_name.upper():
                        return value
                    i += 1
                except OSError:
                    break
    except Exception:
        pass
    return ""


def _get_env(key: str, default: str = "") -> str:
    """
    安全获取环境变量

    优先级:
    1. 系统环境变量 (os.environ)   ← 当前会话
    2. Windows 注册表               ← setx 写入但尚未生效
    3. .env 文件补充值              ← 仅本地开发
    4. 内置默认值
    """
    # 1. 系统环境变量优先
    val = os.environ.get(key)
    if val is not None and val:
        return val

    # 2. Windows: 从注册表读取（setx 设置的变量对新进程不可见时）
    reg_val = _read_from_registry(key)
    if reg_val:
        return reg_val

    # 3. .env 兜底
    _dotenv = _load_env_file()
    if key in _dotenv:
        return _dotenv[key]

    return default


def load_config(config_path: Optional[str] = None) -> dict:
    """
    加载完整配置

    配置来源（优先级从高到低）:
    1. 系统环境变量              —— 生产部署
    2. .env 文件补充值           —— 本地开发
    3. config/settings.yaml     —— 非敏感默认值
    4. 代码内置默认值             —— 最终兜底

    Returns:
        dict: 完整配置字典
    """
    config = dict(_DEFAULTS)

    # 1. 尝试加载 YAML（仅非敏感默认值）
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                _flatten_update(config, yaml_config)
        except Exception as e:
            logger.warning(f"YAML配置加载失败: {e}")

    # 2. 从环境变量补充敏感值（按顺序查找每个可能的变量名）
    for config_key, env_var_names in _SECRET_ENV_MAP.items():
        for env_var in env_var_names:
            env_val = _get_env(env_var)
            if env_val and "placeholder" not in env_val.lower() and len(env_val) > 10:
                # 使用扁平键存储，与 _DEFAULTS 一致的格式
                config[config_key] = env_val
                break

    # 3. .env 兜底值
    dotenv = _load_env_file()
    for config_key, env_var_names in _SECRET_ENV_MAP.items():
        for env_var in env_var_names:
            if env_var in dotenv and env_var not in os.environ:
                env_val = dotenv[env_var]
                if env_val and "placeholder" not in env_val.lower() and len(env_val) > 10:
                    config[config_key] = env_val
                    break

    return config


# ============================================================
# 快捷访问
# ============================================================

def get_llm_config() -> dict:
    """获取 LLM 配置"""
    cfg = load_config()
    return {
        "provider":    cfg.get("llm.provider", "deepseek"),
        "api_key":     cfg.get("llm.api_key", ""),
        "model":       cfg.get("llm.model", "deepseek-chat"),
        "base_url":    cfg.get("llm.base_url", "https://api.deepseek.com/v1"),
        "temperature": cfg.get("llm.temperature", 0.1),
        "max_tokens":  cfg.get("llm.max_tokens", 4096),
        "timeout":     cfg.get("llm.timeout", 60),
        "max_retries": cfg.get("llm.max_retries", 2),
    }


def get_server_config() -> dict:
    """获取服务器配置"""
    cfg = load_config()
    return {
        "host": cfg.get("server.host", "0.0.0.0"),
        "port": cfg.get("server.port", 8000),
    }


def get_data_config() -> dict:
    """获取数据配置"""
    cfg = load_config()
    return {
        "upload_dir": cfg.get("data.upload_dir", "./data/uploads"),
        "max_file_size_mb": cfg.get("data.max_file_size_mb", 50),
    }


def check_api_key() -> bool:
    """
    检查 API Key 是否已配置

    Returns:
        True 如果至少有一个有效的 API Key
    """
    for _config_key, env_var_names in _SECRET_ENV_MAP.items():
        for env_var in env_var_names:
            val = _get_env(env_var)
            if val and "placeholder" not in val.lower() and len(val) > 10:
                return True
    return False


# ============================================================
# 内部工具
# ============================================================

def _set_nested(d: dict, keys: list, value: Any):
    """设置嵌套字典值: d['a.b.c'] → d['a']['b']['c']"""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _flatten_update(target: dict, source: dict, prefix: str = ""):
    """将嵌套 YAML 配置扁平化更新到 target"""
    for key, value in source.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not any(
            isinstance(v, (dict, list)) for v in value.values()
        ):
            # 叶子节点dict → 逐个设置
            for sub_key, sub_val in value.items():
                _set_nested(target, f"{full_key}.{sub_key}".split("."), sub_val)
        elif isinstance(value, dict):
            _flatten_update(target, value, full_key)
        else:
            _set_nested(target, full_key.split("."), value)
