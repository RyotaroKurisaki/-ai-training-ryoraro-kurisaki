from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

def build_parser() -> argparse.ArgumentParser:
    """Day03のCLI引数を定義します（要件文とリトライ回数）。"""
    p = argparse.ArgumentParser(prog="day03")
    p.add_argument("--requirements", required=True)
    p.add_argument("--max-retry", type=int, default=1)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.requirements:
        raise ValueError("--requirements is required")
    if not (0 <= args.max_retry <= 3):
        raise ValueError("--max-retry must be between 0 and 3")


def generate_json(requirements: str) -> str:
    """要件文字列から、JSON文字列（本文のみ）を生成して返します。

    この関数を実装すると、`python -m day03.app --requirements ...` が動くようになります。

    実装ガイド：
    - LLMに「JSONだけを返す」ように強く指示する
    - `title` / `tasks` / `risks` を必ず含める
    - `tasks` は配列で、各要素に `id` / `description` / `acceptance_criteria` を含める
    - 返す文字列は JSON として `json.loads()` できる必要がある

    注意：
    - 余計な前置き/後置きの文章を混ぜない
    - 壊れやすいので、プロンプトは短く・形式を固定する
    """
    load_dotenv()

    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not region:
        raise ValueError("AWS_REGION is required")
    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID is required")

    prompt = f"""
あなたは業務要件を整理するアシスタントです。
以下の要件を分析し、指定されたJSON形式で出力してください。

要件：
{requirements}

出力ルール：
- JSONのみを出力すること。説明文やMarkdownのコードブロックは禁止。
- titleは要件を簡潔に要約したタイトルとする。
- tasksは作業内容を整理した配列とする。
- 各taskにはid（連番の整数）、description、acceptance_criteriaを含める。
- risksは想定されるリスクを格納した文字列の配列とする。

出力形式：
{{
  "title": "タイトル",
  "tasks": [
    {{
      "id": 1,
      "description": "作業内容",
      "acceptance_criteria": "完了条件"
    }}
  ],
  "risks": [
    "想定リスク"
  ]
}}
"""

    config = Config(
        connect_timeout=30,
        read_timeout=30,
    )

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=config,
    )

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    raw_body = response["body"].read()
    logging.info("Bedrock response body: %r", raw_body[:500])

    response_body = json.loads(raw_body)

    reply = "".join(
        item.get("text", "")
        for item in response_body["content"]
        if item.get("type") == "text"
    )

    reply = reply.strip()

    if reply.startswith("```"):
        lines = reply.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        reply = "\n".join(lines).strip()

    logging.info("Generated JSON text: %r", reply[:500])

    return reply

def validate_json(text: str) -> Dict[str, Any]:
    """生成結果のJSONを検証します（必須キーと型）。"""
    obj = json.loads(text)
    for key in ("title", "tasks", "risks"):
        if key not in obj:
            raise ValueError(f"missing key: {key}")
    if not isinstance(obj.get("tasks"), list):
        raise ValueError("tasks must be a list")
    return obj


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    JSON生成→検証→（失敗時は再生成）までを制御します。受講者は `generate_json()` を実装します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    last_err: Exception | None = None
    for _ in range(args.max_retry + 1):
        try:
            text = generate_json(args.requirements)
            validate_json(text)
            print(text)
            return 0
        except NotImplementedError as e:
            logging.error(str(e))
            print(str(e), file=sys.stderr)
            return 1
        except Exception as e:
            last_err = e

    msg = str(last_err) if last_err else "validation failed"
    logging.error(msg)
    print(msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
