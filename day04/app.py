from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from datetime import date

from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

def build_parser() -> argparse.ArgumentParser:
    """Day04のCLI引数を定義します（ユーザー入力テキスト）。"""
    p = argparse.ArgumentParser(prog="day04")
    p.add_argument("--text", required=True)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.text:
        raise ValueError("--text is required")

@tool
def today() -> str:
    """今日の日付をYYYY-MM-DD形式で取得します。"""
    logging.info("Tool called: today")

    return date.today().isoformat()

def validate_tool_args(tool_call: dict) -> None:
    """ツール名と引数を検証する。"""

    if tool_call["name"] != "today":
        raise ValueError(
            f"Unsupported tool: {tool_call['name']}"
        )

    if tool_call.get("args", {}) != {}:
        raise ValueError("today does not accept arguments")
    
def run_chain(text: str) -> str:
    """LangChain + Tool calling を使って回答（文字列）を返します。"""

    import os
    from dotenv import load_dotenv

    load_dotenv()

    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not region:
        raise ValueError("AWS_REGION is required")

    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID is required")

    # AWS BedrockのLLMを初期化
    llm = ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={
            "temperature": 0.2,
            "max_tokens": 512,
        },
    )

    # LLMが利用できるツールを登録
    tools = [today]

    # ツールをLLMにバインド
    llm_with_tools = llm.bind_tools(tools)
    # ユーザーの入力をLLMに渡す
    messages = [
        HumanMessage(
            content=f"""
あなたは日付を回答するアシスタントです。
今日の日付を聞かれた場合は、必ずtodayツールを使用してください。

ユーザーの質問：
{text}
"""
        )
    ]

    # LLMにツールを選択させる
    response = llm_with_tools.invoke(messages)

    # ツール呼び出しの内容を取得
    tool_calls = response.tool_calls

    if not tool_calls:
        raise ValueError("LLM did not call any tool")

    # 許可されたツールだけを実行
    results = []

    for tool_call in tool_calls:
        if tool_call["name"] != "today":
            raise ValueError(
                f"Unsupported tool: {tool_call['name']}"
            )

        # todayは引数なしのため、引数があれば実行しない
        if tool_call.get("args", {}) != {}:
            raise ValueError("today does not accept arguments")

        logging.info("Executing tool: today")

        result = today.invoke({})
        results.append(str(result))

    # ツールの実行結果をLLMに渡す
    final_messages = messages + [
        response,
    ]

    for tool_call, result in zip(tool_calls, results):
        from langchain_core.messages import ToolMessage

        final_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )
        )

    # LLMに最終回答を生成させる
    final_response = llm.invoke(final_messages)

    return str(final_response.content)

def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `run_chain()` の実装に集中し、ここは原則編集しません。
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

    try:
        out = run_chain(args.text)
        print(out)
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
