from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    """Day02のCLI引数を定義します（READMEの機能要件に対応）。"""
    p = argparse.ArgumentParser(prog="day02")
    p.add_argument("--prompt", required=True)
    p.add_argument("--region", default=None)
    p.add_argument("--model-id", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout-sec", type=int, default=30)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.prompt:
        raise ValueError("--prompt is required")
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("--temperature must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be a positive integer")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be a positive integer")


def invoke_bedrock(
    *,
    prompt: str,
    region: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    """Bedrockを呼び出して回答本文（文字列）を返します。

    この関数を実装すると、`python -m day02.app ...` が動くようになります。

    実装ガイド：
    - boto3のBedrock Runtimeクライアントを作る（リージョンは `region` を使う）
    - `model_id` で指定されたモデルを呼び出す
    - `temperature` / `max_tokens` をリクエストに反映する
    - `timeout_sec` はHTTPクライアント設定やタイムアウト制御に反映する
    - 返すのは「回答本文のみ」（前後に装飾文を混ぜない）

    エラー時：
    - 認証/権限/ネットワーク/タイムアウトなどは例外として投げてOK
     （main側で終了コード=1にしてstderrへ出ます）
    """
    try:
        # HTTP通信のタイムアウト設定
        config = Config(
            connect_timeout=timeout_sec,
            read_timeout=timeout_sec,
        )

        # Bedrock Runtimeクライアントを作成
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=config,
        )

        # Claudeに送信するリクエストを作成
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        # Bedrock経由でClaudeを呼び出す
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        # レスポンスをJSONとして読み込む
        response_body = json.loads(response["body"].read())

        # 回答本文だけを取り出す
        reply = "".join(
            item.get("text", "")
            for item in response_body["content"]
            if item.get("type") == "text"
        )

        if not reply:
            raise ValueError("Bedrock returned an empty response")

        return reply

    except Exception as e:
        logging.error(
            "Bedrock invocation failed: %s: %s",
            type(e).__name__,
            e,
        )
        raise


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は原則 `invoke_bedrock()` のみ実装し、それ以外は触らない想定です。
    """
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    region: Optional[str] = args.region or os.getenv("AWS_REGION")
    model_id: Optional[str] = args.model_id or os.getenv("BEDROCK_MODEL_ID")

    if not region:
        msg = "region is required: set --region or AWS_REGION"
        logging.error(msg)
        print(msg, file=sys.stderr)
        return 2

    if not model_id:
        msg = "model-id is required: set --model-id or BEDROCK_MODEL_ID"
        logging.error(msg)
        print(msg, file=sys.stderr)
        return 2

    logging.info(
        "region=%s model-id=%s temperature=%s max-tokens=%s timeout-sec=%s",
        region,
        model_id,
        args.temperature,
        args.max_tokens,
        args.timeout_sec,
    )

    try:
        reply = invoke_bedrock(
            prompt=args.prompt,
            region=region,
            model_id=model_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
        )
        print(reply)
        return 0
    except NotImplementedError as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        logging.error("%s", e)
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
