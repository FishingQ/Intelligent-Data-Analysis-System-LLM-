"""
九天·梧桐云平台连通性测试

用法（先设置环境变量，再运行）:
    setx JIUTIAN_APP_CODE "你的AppCode"
    setx LLM_PROVIDER jiutian
    python scripts/test_jiutian.py

验证: AppCode 鉴权 + generate 接口是否连通、响应能否解析
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_llm_config
from backend.llm.client import create_llm_client


def main():
    cfg = get_llm_config()
    print("=" * 50)
    print("  provider :", cfg["provider"])
    print("  base_url :", cfg["base_url"])
    print("  app_code :", "已配置" if cfg.get("app_code") else "未配置（请 setx JIUTIAN_APP_CODE）")
    print("=" * 50)

    if not cfg.get("app_code"):
        print("[失败] 未检测到 AppCode，请先运行:")
        print('  setx JIUTIAN_APP_CODE "你的AppCode"')
        print("  然后重启终端再运行本脚本。")
        return 1

    client = create_llm_client(
        provider=cfg["provider"],
        api_key=cfg["api_key"],
        app_code=cfg["app_code"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        timeout=cfg["timeout"],
    )
    print("客户端类型:", type(client).__name__)

    try:
        answer = client.chat("你好，请用一句话介绍你自己")
        print("\n[回答] ", answer)
        print("\n[成功] 九天接口连通，响应解析正常。")
        return 0
    except Exception as e:
        print("\n[失败] 调用九天接口出错:")
        print(f"  {e}")
        print("\n若为 401/鉴权错误，请检查 AppCode 是否正确；")
        print("若为 404/连接错误，请检查 base_url 是否为平台给出的 generate 地址前缀。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
