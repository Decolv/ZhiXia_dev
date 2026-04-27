"""测试云端 API Key 是否可用

用法:
    python test_cloud_api.py
    # 或指定参数
    python test_cloud_api.py --url https://api.moonshot.cn/v1/chat/completions --key YOUR_KEY --model moonshot-v1-8k
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Tuple


def test_api(base_url: str, api_key: str, model: str) -> Tuple[bool, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "你好，请只回复'测试成功'四个字。"}],
        "max_tokens": 10,
        "temperature": 0.1,
        "stream": False,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base_url, data=data, headers=headers, method="POST")

    print(f"\n测试接口: {base_url}")
    print(f"模型: {model}")
    print(f"API Key: {api_key[:12]}...{api_key[-4:]}")
    print("-" * 50)

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            print("[OK] 连接成功！")
            print(f"回复内容: {content.strip()}")
            print(f"模型: {result.get('model', 'unknown')}")
            print(f"用量: {json.dumps(result.get('usage', {}), ensure_ascii=False)}")
            return True, model
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[FAIL] HTTP {e.code}")
        try:
            err_json = json.loads(error_body)
            print(json.dumps(err_json, indent=2, ensure_ascii=False))
            err_msg = err_json.get("error", {}).get("message", "")
            err_type = err_json.get("error", {}).get("type", "")
            if err_type or err_msg:
                return False, f"{err_type}: {err_msg}"
        except Exception:
            print(error_body[:500])
        return False, f"HTTP {e.code}"
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="测试云端 LLM API 连通性")
    parser.add_argument("--url", default=os.environ.get("TEST_API_URL", "https://api.openai.com/v1/chat/completions"), help="API 接口地址")
    parser.add_argument("--key", default=os.environ.get("TEST_API_KEY", ""), help="API Key")
    parser.add_argument("--model", default=os.environ.get("TEST_API_MODEL", "gpt-3.5-turbo"), help="模型名称")
    parser.add_argument("--models", default="", help="逗号分隔的候选模型列表，依次尝试")
    args = parser.parse_args()

    if not args.key:
        print("错误: 请通过 --key 参数或 TEST_API_KEY 环境变量提供 API Key")
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [args.model]

    print("=" * 50)
    print("云端 LLM API 连通性测试")
    print("=" * 50)

    success = False
    working_model = None
    for model in models:
        ok, info = test_api(args.url, args.key, model)
        if ok:
            success = True
            working_model = model
            break

    print("\n" + "=" * 50)
    if success:
        print(f"结果: API Key 有效，可用模型: {working_model}")
        print("=" * 50)
        cfg = {
            "llm": {
                "enable_cloud_fallback": True,
                "cloud_api_url": args.url,
                "cloud_api_key": args.key,
                "cloud_model_name": working_model,
                "max_new_tokens": 256,
                "temperature": 0.8,
                "top_p": 0.95,
            }
        }
        print("\n推荐项目配置片段:")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        sys.exit(0)
    else:
        print(f"结果: 测试失败 ({info})")
        sys.exit(1)


if __name__ == "__main__":
    main()
