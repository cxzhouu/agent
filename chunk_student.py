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
    学生文档 id 统一转为更稳定格式。
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
# 3. 学生姓名 / 文档标题推断
# =========================
def infer_student_name(sections: List[Section], source_file: Path, doc_id: str) -> str:
    """
    优先从“姓名”字段提取学生名。
    提取不到时退回一级标题或 doc_id。
    """
    for sec in sections:
        if sec.title.strip() == "姓名":
            value = normalize_text(sec.content)
            if value:
                return value

    for sec in sections:
        if sec.level == 1 and sec.title.strip():
            return sec.title.strip()

    if doc_id:
        return doc_id

    return source_file.stem


def infer_doc_title(sections: List[Section], source_file: Path, student_name: str) -> str:
    if sections and sections[0].title.strip():
        return sections[0].title.strip()
    if student_name:
        return student_name
    return source_file.stem


# =========================
# 4. chunk 类型识别
# =========================
def infer_chunk_type(section: Section) -> str:
    """
    更适合学生档案的知识块类型。
    """
    path_text = " / ".join(section.path)
    title_text = section.title
    full_text = f"{path_text} / {title_text}"

    type_rules = [
        (r"基本信息|姓名|年级|专业|身份信息|个人信息", "basic_profile"),
        (r"身份定位|当前身份|自我认知", "self_identity"),
        (r"性格与表达风格|性格特点|表达风格|说话风格", "persona_trait"),
        (r"兴趣与能力基础|兴趣方向|当前擅长|当前薄弱点|能力基础", "ability_profile"),
        (r"学习状态|当前学习阶段|当前学习目标|当前学习困惑", "learning_state"),
        (r"价值取向|认可的价值观|关注的问题|成长期待|对未来的期待", "value_orientation"),
        (r"对话行为特征|常见提问方式|适合的回应方式|提问习惯", "dialogue_preference"),
        (r"首轮对话生成信息|面向英雄提问时的身份介绍|当前最想解决的问题|希望从英雄身上得到的启发", "opening_prompt"),
        (r"学习记录区|当前阶段总结|已形成的阶段成果|阶段反思|课堂感悟", "learning_record"),
        (r"\bQA\b|\bQ&A\b|问答|问题回答|问题解答", "qa_reference"),
    ]

    for pattern, chunk_type in type_rules:
        if re.search(pattern, full_text, re.I):
            return chunk_type

    return "general"


# =========================
# 5. 拆分逻辑
# =========================
def split_paragraph_into_sentences(paragraph: str, max_chars: int = 220) -> List[str]:
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


def split_long_text(text: str, max_chars: int = 360) -> List[str]:
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


def split_list_like_items(text: str, max_chars: int = 240) -> List[str]:
    """
    把列表型内容拆成更适合检索的小块。
    """
    text = normalize_text(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    new_item_pattern = re.compile(
        r"^(?:[-*•●▪■]|（?[0-9一二三四五六七八九十]+[）.、])"
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


def split_qa_section(text: str) -> List[Dict[str, str]]:
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
def build_tags(section: Section, chunk_type: str, student_name: str) -> List[str]:
    tags = set()
    tags.add(chunk_type)

    if student_name:
        tags.add(student_name)

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

    student_name = infer_student_name(sections, source_file=source_file, doc_id=doc_id)
    doc_title = infer_doc_title(sections, source_file=source_file, student_name=student_name)

    for sec in sections:
        if not sec.content.strip():
            continue

        chunk_type = infer_chunk_type(sec)
        base_meta = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "student_name": student_name,
            "entity_name": student_name,  # 为了兼容后续 embed / rag 可复用字段
            "section_path": sec.path,
            "source_file": str(source_file),
            "tags": build_tags(sec, chunk_type, student_name),
        }

        # 问答
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
                parts = split_long_text(sec.content, max_chars=320)
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

        # 列表型内容：性格、兴趣、困惑、提问方式、回应偏好等
        if chunk_type in {
            "persona_trait",
            "ability_profile",
            "learning_state",
            "value_orientation",
            "dialogue_preference",
            "opening_prompt",
            "learning_record",
        }:
            items = split_list_like_items(sec.content, max_chars=240)
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

        # 其余档案类内容
        max_chars_map = {
            "basic_profile": 240,
            "self_identity": 260,
            "general": 300,
        }

        parts = split_long_text(sec.content, max_chars=max_chars_map.get(chunk_type, 300))
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
# 8. 单文件处理
# =========================
def save_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def process_one_markdown(input_file: Path, output_root: Path, doc_id: Optional[str] = None) -> Dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(f"未找到 Markdown 文件：{input_file}")

    final_doc_id = slugify_doc_id(doc_id or input_file.stem)

    md_text = input_file.read_text(encoding="utf-8")
    sections = parse_markdown(md_text)
    chunks = chunk_sections(sections, source_file=input_file, doc_id=final_doc_id)

    output_dir = output_root / final_doc_id
    output_file = output_dir / f"{final_doc_id}_chunks.jsonl"
    save_jsonl(chunks, output_file)

    student_name = infer_student_name(sections, source_file=input_file, doc_id=final_doc_id)

    return {
        "doc_id": final_doc_id,
        "student_name": student_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "section_count": len(sections),
        "chunk_count": len(chunks),
    }


# =========================
# 9. 批量处理学生档案
# =========================
def process_student_markdowns(raw_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    results = []
    md_files = sorted(raw_root.glob("*.md"))

    for input_file in md_files:
        result = process_one_markdown(
            input_file=input_file,
            output_root=output_root,
            doc_id=input_file.stem
        )
        results.append(result)

    return results


# =========================
# 10. 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="学生档案 Markdown -> chunk JSONL")

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="单个学生 markdown 文件路径，例如 data/raw/students/student_wanghaoyu.md"
    )
    parser.add_argument(
        "--doc_id",
        type=str,
        default=None,
        help="当前文档的 doc_id，例如 student_wanghaoyu"
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw/students",
        help="批量模式下学生 markdown 根目录，默认 data/raw/students"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/chunk/students",
        help="chunk 输出根目录，默认 data/chunk/students"
    )
    parser.add_argument(
        "--batch_students",
        action="store_true",
        help="批量处理学生档案"
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir

    raw_root = project_root / args.raw_root
    output_root = project_root / args.output_root

    if args.batch_students:
        results = process_student_markdowns(raw_root=raw_root, output_root=output_root)
        print("===== 批量处理完成 =====")
        for item in results:
            print(
                f"[{item['doc_id']}] "
                f"学生={item['student_name']} | "
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
        print(f"学生名: {result['student_name']}")
        print(f"输入文件: {result['input_file']}")
        print(f"识别 section 数: {result['section_count']}")
        print(f"生成 chunk 数: {result['chunk_count']}")
        print(f"输出文件: {result['output_file']}")
        return

    raise ValueError("请传入 --input_file 或使用 --batch_students")


if __name__ == "__main__":
    main()