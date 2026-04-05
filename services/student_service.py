import json
from pathlib import Path
from typing import Dict, Any, List

from chunk_student import process_one_markdown
from embed_student import process_one_doc as process_student_embedding


class StudentService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.raw_root = self.project_root / "data" / "raw" / "students"
        self.chunk_root = self.project_root / "data" / "chunk" / "students"
        self.embedding_root = self.project_root / "data" / "embedding" / "students"

        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.chunk_root.mkdir(parents=True, exist_ok=True)
        self.embedding_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_list(items: List[str]) -> str:
        valid = [str(x).strip() for x in items if str(x).strip()]
        if not valid:
            return "暂无"
        return "\n".join(f"- {x}" for x in valid)

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return "暂无"
        text = str(value).strip()
        return text if text else "暂无"

    def build_student_doc_id(self, name: str) -> str:
        base = name.strip().lower().replace(" ", "_")
        return f"student_{base}"

    def build_student_markdown(self, payload: Dict[str, Any]) -> str:
        name = self._normalize(payload.get("name"))
        grade = self._normalize(payload.get("grade"))
        major = self._normalize(payload.get("major"))

        identity = self._normalize(payload.get("identity"))
        self_view = self._normalize(payload.get("self_view"))

        personality_traits = self._format_list(payload.get("personality_traits", []))
        speaking_style = self._format_list(payload.get("speaking_style", []))

        interests = self._format_list(payload.get("interests", []))
        strengths = self._format_list(payload.get("strengths", []))
        weaknesses = self._format_list(payload.get("weaknesses", []))

        current_stage = self._normalize(payload.get("current_stage"))
        learning_goals = self._format_list(payload.get("learning_goals", []))
        current_confusions = self._format_list(payload.get("current_confusions", []))

        concerns = self._format_list(payload.get("concerns", []))
        values = self._format_list(payload.get("values", []))
        expectations = self._normalize(payload.get("expectations"))

        common_questions = self._format_list(payload.get("common_questions", []))
        response_preference = self._format_list(payload.get("response_preference", []))

        opening_intro = self._normalize(payload.get("opening_intro"))
        opening_problem = self._normalize(payload.get("opening_problem"))
        opening_expectation = self._normalize(payload.get("opening_expectation"))

        return f"""# 学生智能体档案

## 基本信息

### 姓名
{name}

### 年级
{grade}

### 专业
{major}

## 身份定位

### 当前身份
{identity}

### 自我认知
{self_view}

## 性格与表达风格

### 性格特点
{personality_traits}

### 表达风格
{speaking_style}

## 兴趣与能力基础

### 兴趣方向
{interests}

### 当前擅长
{strengths}

### 当前薄弱点
{weaknesses}

## 学习状态

### 当前学习阶段
{current_stage}

### 当前学习目标
{learning_goals}

### 当前学习困惑
{current_confusions}

## 价值取向与成长期待

### 关注的问题
{concerns}

### 认可的价值观
{values}

### 对未来的期待
{expectations}

## 对话行为特征

### 常见提问方式
{common_questions}

### 适合的回应方式
{response_preference}

## 首轮对话生成信息

### 面向英雄提问时的身份介绍
{opening_intro}

### 当前最想解决的问题
{opening_problem}

### 希望从英雄身上得到的启发
{opening_expectation}

## 学习记录区

### 当前阶段总结
暂无

### 已形成的阶段成果
暂无

### 后续可写入内容
- 成人宣言
- 成才路线图
- 成功定义书
- 阶段反思
- 课堂感悟
"""

    def create_student(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload["name"]
        doc_id = self.build_student_doc_id(name)

        md_text = self.build_student_markdown(payload)
        md_file = self.raw_root / f"{doc_id}.md"
        md_file.write_text(md_text, encoding="utf-8")

        chunk_result = process_one_markdown(
            input_file=md_file,
            output_root=self.chunk_root,
            doc_id=doc_id,
        )

        embedding_result = process_student_embedding(
            doc_id=doc_id,
            project_root=self.project_root,
            chunk_root="data/chunk/students",
            output_root="data/embedding/students",
        )

        return {
            "doc_id": doc_id,
            "name": name,
            "markdown_file": str(md_file),
            "chunk_result": chunk_result,
            "embedding_result": embedding_result,
        }

    def list_students(self) -> List[Dict[str, Any]]:
        results = []
        for md_file in sorted(self.raw_root.glob("*.md")):
            results.append({
                "doc_id": md_file.stem,
                "file": str(md_file),
            })
        return results

    def get_student_markdown(self, doc_id: str) -> str:
        md_file = self.raw_root / f"{doc_id}.md"
        if not md_file.exists():
            raise FileNotFoundError(f"未找到学生档案：{doc_id}")
        return md_file.read_text(encoding="utf-8")