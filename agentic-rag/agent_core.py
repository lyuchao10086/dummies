import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent.parent / ".env")


def build_llm():
    # 本地 qwen3 负责工具决策和生成；Embedding 使用独立模型。
    return ChatOllama(
        model="qwen3:latest",
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=0,
        reasoning=False,
    )


# 这是受控测试数据，不是关于真实产品的事实。
# 子块只保留摘要；答案所需的具体参数仅存在于父块中。
PARENTS = {
    "aurora_p0": {
        "source": "aurora_test.md",
        "content": "Aurora 是测试系统。默认重试次数为 7 次，等待间隔为 13 秒。",
    }
}
CHILDREN = [
    {
        "parent_id": "aurora_p0",
        "source": "aurora_test.md",
        "content": "Aurora 的重试策略见本节，包括次数与间隔。",
    }
]


@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """用搜索返回的父块ID读取完整上下文，不能猜测ID或者重复读取。"""
    parent = PARENTS.get(parent_id)
    if parent is None:
        return "NO_PARENT_DOCUMENT"
    # 和搜索结果保持相同的标签，让来源和父块关系可追踪。
    return (
        f"Parent ID: {parent_id}\t"
        f"File Name: {parent['source']}\t"
        f"Content: {parent['content']}"
    )


@tool
def search_child_chunks(query: str, limit: int = 3) -> str:
    """搜索文档摘要，返回来源和父块 ID。关键词应简短，例如 Aurora。"""
    if limit < 1:
        return "INVALID_LIMIT: limit 必须大于等于 1"
    hits = [c for c in CHILDREN if query.strip().casefold() in c["content"].casefold()]
    if not query.strip() or not hits:
        return "NO_RELEVANT_CHUNKS"
    return "\n".join(
        f"Parent ID: {c['parent_id']}"
        f"File Name: {c['source']}"
        f"Content: {c['content']}"
        for c in hits[:limit]
    )


TOOLS = [search_child_chunks, retrieve_parent_chunks]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}

SYSTEM_PROMPT = """你是文档检索助手，使用简体中文回答。
必须先搜索文档。搜索时优先使用短关键词，例如系统名称。
摘要没有足够事实时，使用搜索返回的 parent_id 读取完整父块。
不要重复读取同一父块。证据足够后停止调用工具，回答并标明文件名。
只依据工具返回的文档证据；没有证据时说明无法从文档确认。
工具返回错误时，不把错误文本当成文档事实。
"""


def build_agent_model(llm):
    # 工具注册与提示词属于 Agent 配置，和供应商连接参数分开。
    # 保持自动工具选择，以便观察模型实际决策，不强制预设调用路径。
    return llm.bind_tools(TOOLS)


def start_messages(question: str):
    # 这里只保存当前任务的输入；跨问题会话记忆在第 6 阶段设计。
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]


def request_model(agent, messages):
    response = agent.invoke(messages)
    # AIMessage 中的工具请求必须保留，之后才能与 ToolMessage 配对。
    # 迁移到 LangGraph 时，这部分成为模型节点的消息更新。
    messages.append(response)
    return response


def execute_tools(calls, messages):
    for call in calls:
        selected = TOOLS_BY_NAME.get(call["name"])
        if selected is None:
            # 工具不在注册表里时，不运行任何模型生成的代码。
            output = f"UNKNOWN_TOOL: {call['name']}"
        else:
            try:
                output = selected.invoke(call["args"])
            except (ValueError, TypeError) as exc:
                # 包括工具输入校验错误；明确返回错误，不当成正常证据。
                output = f"TOOL_INPUT_ERROR: {exc}"

        # 一条响应可能包含多个工具请求，每个都需要对应结果。
        # 不删除请求，不改写 id，不用用户消息冒充工具结果。
        messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        print("工具：", call["name"], "\t参数：", call["args"], "\t结果：", output)


def run_agent(agent, question: str, max_rounds: int = 6, max_tool_calls: int = 8):
    messages = start_messages(question)
    tool_cnt = 0

    for ix in range(max_rounds):
        print(f"模型轮次：{1+ix}")
        response = request_model(agent, messages)
        calls = response.tool_calls

        if not calls:
            #  没有后续工具请求，正常完成；保留轨迹用于检查证据来源。
            return {
                "status": "completed",
                "answer": response.content,
                "messages": messages,
            }
        if tool_cnt + len(calls) > max_tool_calls:
            # 整批拒绝，避免只执行部分请求导致消息配对不完整。
            # 这里直接退出，不携带尚未配对的消息再次请求模型。
            return {
                "status": "tool_budget_exhausted",
                "answer": "",
                "messages": messages,
            }
        execute_tools(calls, messages)
        tool_cnt += len(calls)
    # 循环耗尽属于未完成，不能把最后一次工具请求冒充答案。
    return {"status": "round_budget_exhausted", "answer": "", "messages": messages}


if __name__ == "__main__":
    # 模型仅在实际运行时创建，便于导入工具进行离线验证。
    agent_model = build_agent_model(build_llm())
    result = run_agent(agent_model, "Aurora 默认重试多少次，间隔多久？")
    print("结束状态：", result["status"])
    print("最终回答：", result["answer"])
