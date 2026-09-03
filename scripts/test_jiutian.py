"""九天 generate_stream 接口探针 —— 验证服务连通性与真实响应格式

用法:
    set JIUTIAN_APP_CODE=你的AppCode      (Windows CMD)
    python scripts/test_jiutian.py "你好，介绍一下你自己"

会依次:
    1. 直接 POST 到 generate_stream 打印原始响应 (确认接口真实格式)
    2. 用 JiutianChatModel 走一次完整调用 (验证适配器)
"""
import os
import sys
import json

import httpx

BASE_URL = "http://127.0.0.1:8090/generate_stream"


def main() -> None:
    app_code = os.getenv("JIUTIAN_APP_CODE") or os.getenv("JIUTIAN_API_KEY") or ""
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好"

    if not app_code:
        print("未配置 AppCode: 请先执行  set JIUTIAN_APP_CODE=你的AppCode")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {app_code}", "Content-Type": "application/json"}
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    print(f"POST {BASE_URL}")
    print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
    print("-" * 60)

    try:
        resp = httpx.post(BASE_URL, json=payload, headers=headers, timeout=120)
    except httpx.HTTPError as e:
        print(f"连接失败: {e}")
        print("请确认本地推理服务已启动 (监听 127.0.0.1:8090)")
        sys.exit(1)

    print(f"状态码: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    print("-" * 60)
    print("原始响应 (前 2000 字符):")
    print(resp.text[:2000])
    print("-" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.llm.jiutian_adapter import JiutianChatModel

    model = JiutianChatModel(base_url=BASE_URL, app_code=app_code, model_name="")
    result = model.invoke(prompt)
    print(f"适配器解析结果: {result.content}")


if __name__ == "__main__":
    main()
