import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


# =========================
# 1. 默认配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
EMBED_MODEL = "jina-embeddings-v3"
TASK = "retrieval.passage"
EMBEDDING_TYPE = "float"
NORMALIZED = True
BATCH_SIZE = 32


# =========================
# 2. 基础读写
# =========================
def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件。
    """
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    保存为 JSONL 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_field(value: Any) -> str:
    """
    把不同类型字段统一转成适合拼接的字符串。
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return " > ".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


# =========================
# 3. embedding 文本构造
# =========================
def build_embedding_text(record: Dict[str, Any]) -> str:
    """
    为学生档案知识块构造更适合检索的 embedding 文本。

    目标：
    1. 保留学生身份信息
    2. 保留块类型、路径、标题、标签
    3. 对 opening_prompt / dialogue_preference / learning_state 等块强化表达
    4. 对问答类 chunk 保留 question / answer
    """
    student_name = normalize_field(record.get("student_name"))
    entity_name = normalize_field(record.get("entity_name"))
    doc_title = normalize_field(record.get("doc_title"))
    chunk_type = normalize_field(record.get("chunk_type"))
    section_path = normalize_field(record.get("section_path"))
    title = normalize_field(record.get("title"))
    tags = normalize_field(record.get("tags"))
    content = normalize_field(record.get("content"))
    question = normalize_field(record.get("question"))
    answer = normalize_field(record.get("answer"))

    lines: List[str] = []

    if student_name:
        lines.append(f"学生姓名：{student_name}")

    if entity_name and entity_name != student_name:
        lines.append(f"实体名称：{entity_name}")

    if doc_title:
        lines.append(f"文档标题：{doc_title}")

    if chunk_type:
        lines.append(f"档案类型：{chunk_type}")

    if section_path:
        lines.append(f"章节路径：{section_path}")

    if title:
        lines.append(f"小节标题：{title}")

    if tags:
        lines.append(f"标签：{tags}")

    if question and answer:
        lines.append(f"问题：{question}")
        lines.append(f"回答：{answer}")
    else:
        if content:
            lines.append(f"内容：\n{content}")

    return "\n".join(lines).strip()


# =========================
# 4. API 调用
# =========================
def get_embeddings(
    texts: List[str],
    jina_api_key: str,
    timeout: int = 120
) -> List[List[float]]:
    """
    调用 Jina Embedding API 获取向量。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jina_api_key}",
    }

    payload = {
        "model": EMBED_MODEL,
        "task": TASK,
        "embedding_type": EMBEDDING_TYPE,
        "normalized": NORMALIZED,
        "input": texts,
    }

    response = requests.post(
        JINA_EMBED_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Jina Embedding API 请求失败: {response.status_code}\n{response.text}"
        )

    data = response.json()

    if "data" not in data:
        raise RuntimeError(f"Jina 返回结果异常，缺少 data 字段：{data}")

    vectors = [item["embedding"] for item in data["data"]]
    return vectors


# =========================
# 5. 记录校验与输出结构
# =========================
def validate_record(record: Dict[str, Any]) -> bool:
    """
    校验学生 chunk 记录是否可用于 embedding。
    """
    if not isinstance(record, dict):
        return False

    content = normalize_field(record.get("content"))
    question = normalize_field(record.get("question"))
    answer = normalize_field(record.get("answer"))

    if content:
        return True

    if question and answer:
        return True

    return False


def build_output_record(
    chunk: Dict[str, Any],
    embedding_text: str,
    vector: List[float]
) -> Dict[str, Any]:
    """
    输出记录中保留原始元信息，方便后续 student rag 检索与调试。
    """
    return {
        "chunk_id": chunk.get("chunk_id"),
        "chunk_type": chunk.get("chunk_type"),
        "title": chunk.get("title"),
        "content": chunk.get("content"),
        "order": chunk.get("order"),
        "doc_id": chunk.get("doc_id"),
        "doc_title": chunk.get("doc_title"),
        "student_name": chunk.get("student_name"),
        "entity_name": chunk.get("entity_name"),
        "section_path": chunk.get("section_path"),
        "source_file": chunk.get("source_file"),
        "tags": chunk.get("tags"),
        "question": chunk.get("question"),
        "answer": chunk.get("answer"),
        "embedding_model": EMBED_MODEL,
        "embedding_task": TASK,
        "embedding_text": embedding_text,
        "embedding": vector,
    }


# =========================
# 6. 路径构造
# =========================
def build_input_chunk_file(project_root: Path, chunk_root: str, doc_id: str) -> Path:
    return project_root / chunk_root / doc_id / f"{doc_id}_chunks.jsonl"


def build_output_embedding_file(project_root: Path, output_root: str, doc_id: str) -> Path:
    return project_root / output_root / doc_id / f"{doc_id}_embeddings.jsonl"


# =========================
# 7. 单个学生 embedding 生成
# =========================
def process_one_doc(
    doc_id: str,
    project_root: Path,
    chunk_root: str = "data/chunk/students",
    output_root: str = "data/embedding/students",
    batch_size: int = BATCH_SIZE,
    jina_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    处理单个学生档案的 embedding 生成。
    """
    final_jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
    if not final_jina_api_key:
        raise ValueError("未检测到 JINA_API_KEY，请先设置环境变量。")

    input_file = build_input_chunk_file(project_root, chunk_root, doc_id)
    output_file = build_output_embedding_file(project_root, output_root, doc_id)

    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入 chunk 文件：{input_file}")

    print(f"读取 chunk 文件: {input_file}")
    chunks = load_jsonl(input_file)
    print(f"共读取 {len(chunks)} 条 chunk")

    valid_chunks: List[Dict[str, Any]] = []
    embed_texts: List[str] = []

    for chunk in chunks:
        if not validate_record(chunk):
            continue

        text = build_embedding_text(chunk)
        if not text.strip():
            continue

        valid_chunks.append(chunk)
        embed_texts.append(text)

    print(f"有效 chunk 数: {len(valid_chunks)}")

    if not valid_chunks:
        raise ValueError(f"{doc_id} 没有可用于向量化的有效 chunk，请检查 chunk 文件内容。")

    output_records: List[Dict[str, Any]] = []

    for start in range(0, len(valid_chunks), batch_size):
        end = min(start + batch_size, len(valid_chunks))
        batch_chunks = valid_chunks[start:end]
        batch_texts = embed_texts[start:end]

        print(f"[{doc_id}] 处理 batch: {start} - {end - 1}")

        vectors = get_embeddings(
            texts=batch_texts,
            jina_api_key=final_jina_api_key,
        )

        if len(vectors) != len(batch_chunks):
            raise RuntimeError(
                f"向量数量与 chunk 数量不一致：vectors={len(vectors)}, chunks={len(batch_chunks)}"
            )

        for chunk, text, vector in zip(batch_chunks, batch_texts, vectors):
            output_records.append(build_output_record(chunk, text, vector))

    save_jsonl(output_records, output_file)

    student_name = ""
    if output_records:
        student_name = normalize_field(output_records[0].get("student_name"))

    return {
        "doc_id": doc_id,
        "student_name": student_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "input_chunk_count": len(chunks),
        "valid_chunk_count": len(valid_chunks),
        "output_record_count": len(output_records),
    }


# =========================
# 8. 批量处理学生档案
# =========================
def process_student_docs(
    project_root: Path,
    chunk_root: str = "data/chunk/students",
    output_root: str = "data/embedding/students",
    batch_size: int = BATCH_SIZE,
    jina_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    批量处理所有学生档案 chunk。
    约定：chunk_root 下每个一级目录名就是一个 student doc_id。
    """
    results = []
    chunk_root_path = project_root / chunk_root

    if not chunk_root_path.exists():
        raise FileNotFoundError(f"未找到学生 chunk 根目录：{chunk_root_path}")

    doc_ids = sorted([p.name for p in chunk_root_path.iterdir() if p.is_dir()])

    for doc_id in doc_ids:
        result = process_one_doc(
            doc_id=doc_id,
            project_root=project_root,
            chunk_root=chunk_root,
            output_root=output_root,
            batch_size=batch_size,
            jina_api_key=jina_api_key,
        )
        results.append(result)

    return results


# =========================
# 9. 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="学生档案 chunk -> embedding 通用脚本")

    parser.add_argument(
        "--doc_id",
        type=str,
        default=None,
        help="单个学生文档 ID，例如 student_wanghaoyu"
    )
    parser.add_argument(
        "--chunk_root",
        type=str,
        default="data/chunk/students",
        help="学生 chunk 根目录，默认 data/chunk/students"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/embedding/students",
        help="学生 embedding 输出根目录，默认 data/embedding/students"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"每批请求条数，默认 {BATCH_SIZE}"
    )
    parser.add_argument(
        "--batch_students",
        action="store_true",
        help="批量处理所有学生档案"
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    jina_api_key = os.getenv("JINA_API_KEY")

    if args.batch_students:
        results = process_student_docs(
            project_root=project_root,
            chunk_root=args.chunk_root,
            output_root=args.output_root,
            batch_size=args.batch_size,
            jina_api_key=jina_api_key,
        )

        print("===== 批量学生 embedding 生成完成 =====")
        for item in results:
            print(
                f"[{item['doc_id']}] "
                f"学生={item['student_name']} | "
                f"输入 chunk={item['input_chunk_count']} | "
                f"有效 chunk={item['valid_chunk_count']} | "
                f"输出 embedding={item['output_record_count']} | "
                f"文件={item['output_file']}"
            )
        return

    if args.doc_id:
        result = process_one_doc(
            doc_id=args.doc_id,
            project_root=project_root,
            chunk_root=args.chunk_root,
            output_root=args.output_root,
            batch_size=args.batch_size,
            jina_api_key=jina_api_key,
        )

        print("===== 单个学生 embedding 生成完成 =====")
        print(f"doc_id: {result['doc_id']}")
        print(f"学生名: {result['student_name']}")
        print(f"输入文件: {result['input_file']}")
        print(f"输入 chunk 数: {result['input_chunk_count']}")
        print(f"有效 chunk 数: {result['valid_chunk_count']}")
        print(f"输出记录数: {result['output_record_count']}")
        print(f"输出文件: {result['output_file']}")
        return

    raise ValueError("请传入 --doc_id 或使用 --batch_students")


if __name__ == "__main__":
    main()