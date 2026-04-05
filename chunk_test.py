import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any


@dataclass
class Section:
    level: int
    title: str
    path: List[str]
    content: str


# =========================
# 1. 基础文本处理
# =========================
def normalize_text(text: str) -> str:
    """
    统一文本格式，减少空白噪声。
    """
    if text is None:
        return ""

    text = text.replace("\u3000", " ")
    text = text.strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def slugify_doc_id(name: str) -> str:
    """
    将输入名转成更稳定的 doc_id。
    规则：
    1. 中文、英文、数字保留
    2. 空格和连接符统一成下划线
    3. 全部转小写
    """
    name = normalize_text(name)
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


# =========================
# 2. Markdown 解析
# =========================
def parse_markdown(md_text: str) -> List[Section]:
    """
    按 Markdown 标题层级解析文档，保留每一节的标题路径。
    """
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    lines = md_text.splitlines()

    sections: List[Section] = []
    stack: List[tuple[int, str]] = []
    current_title: Optional[str] = None
    current_level: Optional[int] = None
    current_content: List[str] = []

    def flush_current():
        nonlocal current_title, current_level, current_content
        if current_title is not None:
            sections.append(
                Section(
                    level=current_level if current_level is not None else 1,
                    title=current_title,
                    path=[title for _, title in stack],
                    content=normalize_text("\n".join(current_content)),
                )
            )

    for line in lines:
        m = heading_pattern.match(line)
        if m:
            flush_current()

            level = len(m.group(1))
            title = m.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            current_title = title
            current_level = level
            current_content = []
        else:
            if current_title is not None:
                current_content.append(line)

    flush_current()
    return sections


# =========================
# 3. 人物名 / 文档名推断
# =========================
def infer_entity_name(sections: List[Section], source_file: Path, doc_id: str) -> str:
    """
    优先用一级标题作为人物名；如果没有，就退回到 doc_id / 文件名。
    """
    for sec in sections:
        if sec.level == 1 and sec.title.strip():
            return sec.title.strip()

    if doc_id:
        return doc_id

    return source_file.stem


def infer_doc_title(sections: List[Section], source_file: Path, entity_name: str) -> str:
    """
    推断文档标题。
    """
    if sections and sections[0].title.strip():
        return sections[0].title.strip()

    if entity_name:
        return entity_name

    return source_file.stem


# =========================
# 4. chunk 类型识别
# =========================
def infer_chunk_type(section: Section) -> str:
    """
    根据标题路径推断人物知识块类型。
    """
    path_text = " / ".join(section.path)
    title_text = section.title
    full_text = f"{path_text} / {title_text}"

    type_rules = [
        (r"人物简介|基本信息|身份简介|生平简介|人物概述|人物档案|简介", "basic_info"),
        (r"生平|人生经历|成长经历|求学经历|革命道路|人生道路|早年经历|留学经历|履历", "life_stage"),
        (r"大事记|时间线|年表|关键事件|重要事件|主要事迹|革命实践|历史事件|事件", "timeline_event"),
        (r"思想|主张|理论|观点|理念|政治主张|学术思想|核心思想", "thought"),
        (r"精神|品质|品格|风范|作风|信念|革命精神|优秀品格", "spirit"),
        (r"语录|名言|名句|经典话语|经典语录|原话", "quote"),
        (r"历史地位|历史贡献|历史评价|人物评价|社会影响|历史影响|贡献|评价|地位", "historical_role"),
        (r"时代价值|现实启示|当代启示|现实意义|青年启示|青年成长|思政结合|育人价值", "era_value"),
        (r"课堂场景|教学场景|对话场景|第[一二三四五六七八九十0-9]+学时|课堂任务|情境导入|成果落地", "classroom_scene"),
        (r"行动建议|行动指引|实践建议|成长建议|路径建议|怎么做|可执行建议|落实路径", "action_guidance"),
        (r"教师引导|课堂引导|教学设计|教学提示|讨论题|引导问题|教师提问|教学应用", "teacher_guide"),
        (r"学生提问|互动话题|对话参考|提问参考|交流话题|常见问题", "student_topic"),
        (r"\bQA\b|\bQ&A\b|问答|问题回答|问题解答|回答示例", "qa_reference"),
    ]

    for pattern, chunk_type in type_rules:
        if re.search(pattern, full_text, re.I):
            return chunk_type

    return "general"


# =========================
# 5. 拆分逻辑
# =========================
def split_paragraph_into_sentences(paragraph: str, max_chars: int = 220) -> List[str]:
    """
    当单个自然段过长时，继续按句子切。
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[。！？；!?;])", paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [paragraph]

    chunks: List[str] = []
    buf: List[str] = []

    for sent in sentences:
        candidate = "".join(buf + [sent])
        if len(candidate) <= max_chars:
            buf.append(sent)
        else:
            if buf:
                chunks.append("".join(buf).strip())
            buf = [sent]

    if buf:
        chunks.append("".join(buf).strip())

    return chunks


def split_long_text(text: str, max_chars: int = 420) -> List[str]:
    """
    先按自然段聚合，再在必要时按句子继续切分。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalize_text(text)) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    buf: List[str] = []

    for p in paragraphs:
        if len(p) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf).strip())
                buf = []
            chunks.extend(split_paragraph_into_sentences(p, max_chars=max_chars))
            continue

        candidate = "\n\n".join(buf + [p])
        if len(candidate) <= max_chars:
            buf.append(p)
        else:
            if buf:
                chunks.append("\n\n".join(buf).strip())
            buf = [p]

    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks


def split_list_like_items(text: str, max_chars: int = 280) -> List[str]:
    """
    优先按项目符号、序号、年份切成条目，适合时间线、教学提示、互动话题。
    """
    text = normalize_text(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    new_item_pattern = re.compile(
        r"^(?:[-*•●▪■]|（?[0-9一二三四五六七八九十]+[）.、]|[0-9]{4}年|[0-9]{4}[./-][0-9]{1,2})"
    )

    items: List[str] = []
    current: List[str] = []

    for line in lines:
        if new_item_pattern.match(line):
            if current:
                items.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        items.append("\n".join(current).strip())

    if len(items) <= 1:
        return split_long_text(text, max_chars=max_chars)

    final_items: List[str] = []
    for item in items:
        if len(item) <= max_chars:
            final_items.append(item)
        else:
            final_items.extend(split_long_text(item, max_chars=max_chars))

    return final_items


def split_quote_section(text: str) -> List[str]:
    """
    语录区尽量做到“一条语录一个 chunk”。
    """
    text = normalize_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    quotes: List[str] = []

    for p in paragraphs:
        lines = [line.strip(">-•●▪■ ").strip() for line in p.split("\n") if line.strip()]
        if len(lines) >= 2 and all(len(line) <= 80 for line in lines):
            quotes.extend(lines)
        else:
            if len(p) <= 180:
                quotes.append(p)
            else:
                quotes.extend(split_long_text(p, max_chars=180))

    return [q for q in quotes if q]


def split_qa_section(text: str) -> List[Dict[str, str]]:
    """
    解析常见问答格式：
    问题：...
    回答：...
    或
    Q: ...
    A: ...
    """
    text = normalize_text(text)
    pattern = re.compile(
        r"(?:^|\n)(?:Q[:：]|问题[:：]|问[:：])\s*(.+?)\n(?:A[:：]|回答[:：]|答[:：])\s*(.+?)(?=\n(?:Q[:：]|问题[:：]|问[:：])|\Z)",
        re.S,
    )
    pairs = []

    for m in pattern.finditer(text):
        question = normalize_text(m.group(1))
        answer = normalize_text(m.group(2))
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


# =========================
# 6. 标签与记录构造
# =========================
def build_tags(section: Section, chunk_type: str, entity_name: str) -> List[str]:
    """
    给后续检索预留标签信息。
    """
    tags = set()
    tags.add(chunk_type)

    if entity_name:
        tags.add(entity_name)

    for item in section.path:
        item = item.strip()
        if item:
            tags.add(item)

    if section.title.strip():
        tags.add(section.title.strip())

    return sorted(tags)


def make_chunk(
    *,
    doc_id: str,
    order: int,
    chunk_type: str,
    title: str,
    content: str,
    base_meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    record = {
        "chunk_id": f"{doc_id}_{order:03d}",
        "chunk_type": chunk_type,
        "title": title,
        "content": content,
        "order": order,
        **base_meta,
    }
    if extra:
        record.update(extra)
    return record


# =========================
# 7. 核心切块
# =========================
def chunk_sections(sections: List[Section], source_file: Path, doc_id: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    order = 0

    entity_name = infer_entity_name(sections, source_file=source_file, doc_id=doc_id)
    doc_title = infer_doc_title(sections, source_file=source_file, entity_name=entity_name)

    for sec in sections:
        if not sec.content.strip():
            continue

        chunk_type = infer_chunk_type(sec)
        base_meta = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "entity_name": entity_name,
            "section_path": sec.path,
            "source_file": str(source_file),
            "tags": build_tags(sec, chunk_type, entity_name),
        }

        # 时间线 / 事迹：尽量一条事件一个 chunk
        if chunk_type == "timeline_event":
            items = split_list_like_items(sec.content, max_chars=260)
            for idx, item in enumerate(items, 1):
                order += 1
                chunks.append(
                    make_chunk(
                        doc_id=doc_id,
                        order=order,
                        chunk_type=chunk_type,
                        title=f"{sec.title} - 事件{idx}",
                        content=item,
                        base_meta=base_meta,
                    )
                )
            continue

        # 语录：尽量一条语录一个 chunk
        if chunk_type == "quote":
            quotes = split_quote_section(sec.content)
            for idx, quote in enumerate(quotes, 1):
                order += 1
                chunks.append(
                    make_chunk(
                        doc_id=doc_id,
                        order=order,
                        chunk_type=chunk_type,
                        title=f"{sec.title} - 语录{idx}",
                        content=quote,
                        base_meta=base_meta,
                    )
                )
            continue

        # 教师引导 / 学生话题 / 课堂场景 / 行动建议：按条目拆
        if chunk_type in {"teacher_guide", "student_topic", "classroom_scene", "action_guidance"}:
            items = split_list_like_items(sec.content, max_chars=280)
            for idx, item in enumerate(items, 1):
                order += 1
                chunks.append(
                    make_chunk(
                        doc_id=doc_id,
                        order=order,
                        chunk_type=chunk_type,
                        title=f"{sec.title} - 条目{idx}",
                        content=item,
                        base_meta=base_meta,
                    )
                )
            continue

        # 问答参考：尽量拆成一问一答
        if chunk_type == "qa_reference":
            qa_pairs = split_qa_section(sec.content)

            if qa_pairs:
                for idx, qa in enumerate(qa_pairs, 1):
                    order += 1
                    chunks.append(
                        make_chunk(
                            doc_id=doc_id,
                            order=order,
                            chunk_type=chunk_type,
                            title=f"{sec.title} - 问答{idx}",
                            content=f"问题：{qa['question']}\n回答：{qa['answer']}",
                            base_meta=base_meta,
                            extra={
                                "question": qa["question"],
                                "answer": qa["answer"],
                            },
                        )
                    )
            else:
                parts = split_long_text(sec.content, max_chars=360)
                for idx, part in enumerate(parts, 1):
                    order += 1
                    chunks.append(
                        make_chunk(
                            doc_id=doc_id,
                            order=order,
                            chunk_type=chunk_type,
                            title=f"{sec.title} - 片段{idx}",
                            content=part,
                            base_meta=base_meta,
                        )
                    )
            continue

        # 其余知识块：按主题段落切
        max_chars_map = {
            "basic_info": 320,
            "life_stage": 360,
            "thought": 360,
            "spirit": 320,
            "historical_role": 360,
            "era_value": 360,
            "general": 360,
        }

        parts = split_long_text(sec.content, max_chars=max_chars_map.get(chunk_type, 360))
        for idx, part in enumerate(parts, 1):
            order += 1
            chunks.append(
                make_chunk(
                    doc_id=doc_id,
                    order=order,
                    chunk_type=chunk_type,
                    title=f"{sec.title} - 片段{idx}" if len(parts) > 1 else sec.title,
                    content=part,
                    base_meta=base_meta,
                )
            )

    return chunks


# =========================
# 8. 文件读写与单人物处理
# =========================
def save_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def process_one_markdown(input_file: Path, output_root: Path, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    处理单个 markdown 文件。
    """
    if not input_file.exists():
        raise FileNotFoundError(f"未找到 Markdown 文件：{input_file}")

    final_doc_id = slugify_doc_id(doc_id or input_file.stem)

    md_text = input_file.read_text(encoding="utf-8")
    sections = parse_markdown(md_text)
    chunks = chunk_sections(sections, source_file=input_file, doc_id=final_doc_id)

    output_dir = output_root / final_doc_id
    output_file = output_dir / f"{final_doc_id}_chunks.jsonl"
    save_jsonl(chunks, output_file)

    entity_name = infer_entity_name(sections, source_file=input_file, doc_id=final_doc_id)

    return {
        "doc_id": final_doc_id,
        "entity_name": entity_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "section_count": len(sections),
        "chunk_count": len(chunks),
    }


# =========================
# 9. 批量处理 5 个英雄
# =========================
DEFAULT_HERO_FILES = {
    "li_dazhao": "li_dazhao.md",
    "jiao_yulu": "jiao_yulu.md",
    "qian_xuesen": "qian_xuesen.md",
    "huang_danian": "huang_danian.md",
    "huang_wenxiu": "huang_wenxiu.md",
}


def process_default_heroes(raw_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    """
    批量处理默认 5 个英雄人物 markdown。
    """
    results = []
    for doc_id, filename in DEFAULT_HERO_FILES.items():
        input_file = raw_root / filename
        result = process_one_markdown(input_file=input_file, output_root=output_root, doc_id=doc_id)
        results.append(result)
    return results


# =========================
# 10. 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人物知识库 Markdown -> chunk JSONL 通用脚本")

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="单个 markdown 文件路径，例如 data/raw/li_dazhao.md"
    )
    parser.add_argument(
        "--doc_id",
        type=str,
        default=None,
        help="当前文档的 doc_id，例如 li_dazhao；不传则自动从文件名生成"
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw",
        help="批量模式下 markdown 根目录，默认 data/raw"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/chunk",
        help="chunk 输出根目录，默认 data/chunk"
    )
    parser.add_argument(
        "--batch_default_heroes",
        action="store_true",
        help="批量处理默认 5 个英雄人物"
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir

    raw_root = project_root / args.raw_root
    output_root = project_root / args.output_root

    if args.batch_default_heroes:
        results = process_default_heroes(raw_root=raw_root, output_root=output_root)
        print("===== 批量处理完成 =====")
        for item in results:
            print(
                f"[{item['doc_id']}] "
                f"人物={item['entity_name']} | "
                f"section={item['section_count']} | "
                f"chunk={item['chunk_count']} | "
                f"输出={item['output_file']}"
            )
        return

    if args.input_file:
        input_file = project_root / args.input_file
        result = process_one_markdown(
            input_file=input_file,
            output_root=output_root,
            doc_id=args.doc_id,
        )
        print("===== 单文件处理完成 =====")
        print(f"doc_id: {result['doc_id']}")
        print(f"人物名: {result['entity_name']}")
        print(f"输入文件: {result['input_file']}")
        print(f"识别 section 数: {result['section_count']}")
        print(f"生成 chunk 数: {result['chunk_count']}")
        print(f"输出文件: {result['output_file']}")
        return

    raise ValueError("请传入 --input_file 或使用 --batch_default_heroes")


if __name__ == "__main__":
    main()