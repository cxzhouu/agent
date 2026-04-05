import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from openai import OpenAI


# =========================
# 1. 默认配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "data" / "embedding" / "students"


# =========================
# 2. Student RAG Pipeline
# =========================
class StudentRAGPipeline:
    def __init__(
        self,
        embedding_file: Path,
        student_name: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        jina_embed_url: str = "https://api.jina.ai/v1/embeddings",
        jina_model: str = "jina-embeddings-v3",
        query_task: str = "retrieval.query",
        jina_rerank_url: str = "https://api.jina.ai/v1/rerank",
        rerank_model: str = "jina-reranker-v3",
        deepseek_model: str = "deepseek-chat",
        recall_top_k: int = 10,
        rerank_top_n: int = 5,
        use_proxy: bool = True,
    ):
        self.embedding_file = Path(embedding_file)
        self.student_name = student_name

        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")

        self.jina_embed_url = jina_embed_url
        self.jina_model = jina_model
        self.query_task = query_task

        self.jina_rerank_url = jina_rerank_url
        self.rerank_model = rerank_model

        self.deepseek_model = deepseek_model
        self.recall_top_k = recall_top_k
        self.rerank_top_n = rerank_top_n
        self.use_proxy = use_proxy

        if not self.jina_api_key:
            raise ValueError("未检测到 JINA_API_KEY，请先设置环境变量或在初始化时传入。")
        if not self.deepseek_api_key:
            raise ValueError("未检测到 DEEPSEEK_API_KEY，请先设置环境变量或在初始化时传入。")
        if not self.embedding_file.exists():
            raise FileNotFoundError(f"找不到 embedding 文件: {self.embedding_file}")

        self.records = self.load_jsonl(self.embedding_file)

        if not self.student_name:
            self.student_name = self.infer_student_name_from_records(self.records)

        self.deepseek_client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

    # =========================
    # 2.1 基础工具函数
    # =========================
    @staticmethod
    def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    @staticmethod
    def normalize_text(text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, list):
            return " > ".join(str(x).strip() for x in text if str(x).strip())
        if isinstance(text, dict):
            return json.dumps(text, ensure_ascii=False)
        return str(text).strip()

    @staticmethod
    def dot_similarity(vec1: List[float], vec2: List[float]) -> float:
        return float(
            np.dot(
                np.array(vec1, dtype=np.float32),
                np.array(vec2, dtype=np.float32)
            )
        )

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = self.use_proxy
        return session

    @staticmethod
    def infer_student_name_from_records(records: List[Dict[str, Any]]) -> str:
        for rec in records:
            name = rec.get("student_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "该学生"

    # =========================
    # 2.2 查询构造
    # =========================
    def build_query_text(self, task: str) -> str:
        return (
            f"学生：{self.student_name}\n"
            f"任务：{task.strip()}"
        )

    # =========================
    # 2.3 向量检索
    # =========================
    def get_query_embedding(self, query_text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}",
        }

        payload = {
            "model": self.jina_model,
            "task": self.query_task,
            "embedding_type": "float",
            "normalized": True,
            "input": [query_text],
        }

        session = self._create_session()
        response = session.post(
            self.jina_embed_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina query embedding 请求失败: {response.status_code}\n{response.text}"
            )

        data = response.json()
        return data["data"][0]["embedding"]

    def search_top_k(
        self,
        task: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.recall_top_k
        query_text = self.build_query_text(task)
        query_vec = self.get_query_embedding(query_text)

        scored = []
        for rec in self.records:
            emb = rec.get("embedding")
            if not emb:
                continue

            score = self.dot_similarity(query_vec, emb)
            scored.append({
                "score": score,
                "chunk_id": rec.get("chunk_id"),
                "title": rec.get("title"),
                "chunk_type": rec.get("chunk_type"),
                "content": rec.get("content"),
                "section_path": rec.get("section_path"),
                "doc_title": rec.get("doc_title"),
                "student_name": rec.get("student_name"),
                "entity_name": rec.get("entity_name"),
                "tags": rec.get("tags"),
                "question": rec.get("question"),
                "answer": rec.get("answer"),
                "source_file": rec.get("source_file"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # =========================
    # 2.4 Rerank
    # =========================
    def build_rerank_document(self, candidate: Dict[str, Any]) -> str:
        student_name = self.normalize_text(candidate.get("student_name"))
        title = self.normalize_text(candidate.get("title"))
        chunk_type = self.normalize_text(candidate.get("chunk_type"))
        section_path = self.normalize_text(candidate.get("section_path"))
        tags = self.normalize_text(candidate.get("tags"))
        content = self.normalize_text(candidate.get("content"))
        question = self.normalize_text(candidate.get("question"))
        answer = self.normalize_text(candidate.get("answer"))

        lines = []

        if student_name:
            lines.append(f"学生姓名：{student_name}")
        if title:
            lines.append(f"标题：{title}")
        if chunk_type:
            lines.append(f"档案类型：{chunk_type}")
        if section_path:
            lines.append(f"章节路径：{section_path}")
        if tags:
            lines.append(f"标签：{tags}")

        if question and answer:
            lines.append(f"问题：{question}")
            lines.append(f"回答：{answer}")
        else:
            lines.append(f"内容：\n{content}")

        return "\n".join(lines).strip()

    def rerank_with_jina(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        top_n = top_n or self.rerank_top_n

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}",
        }

        documents = [self.build_rerank_document(c) for c in candidates]

        payload = {
            "model": self.rerank_model,
            "query": task,
            "documents": documents,
            "top_n": top_n
        }

        session = self._create_session()
        response = session.post(
            self.jina_rerank_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina Rerank API 请求失败: {response.status_code}\n{response.text}"
            )

        data = response.json()

        reranked = []
        for item in data["results"]:
            idx = item["index"]
            rerank_score = item.get("relevance_score", 0.0)

            cand = dict(candidates[idx])
            cand["rerank_score"] = rerank_score
            reranked.append(cand)

        return reranked

    # =========================
    # 2.5 上下文构造
    # =========================
    def build_context(self, top_chunks: List[Dict[str, Any]]) -> str:
        parts = []

        for ch in top_chunks:
            title = self.normalize_text(ch.get("title"))
            chunk_type = self.normalize_text(ch.get("chunk_type"))
            content = self.normalize_text(ch.get("content"))
            question = self.normalize_text(ch.get("question"))
            answer = self.normalize_text(ch.get("answer"))

            lines = []

            if title:
                lines.append(f"主题：{title}")

            if chunk_type:
                lines.append(f"类型：{chunk_type}")

            if question and answer:
                lines.append(f"相关问答：{question}")
                lines.append(f"参考表达：{answer}")
            else:
                lines.append(f"档案内容：{content}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # =========================
    # 2.6 学生摘要生成
    # =========================
    def build_profile_summary(self, return_chunks: bool = False) -> Any:
        """
        生成给老师智能体使用的学生信息摘要。
        """
        task = (
            "请提取这名学生最关键的身份信息，用于教师理解这个学生。"
            "重点包括：姓名、年级、专业、当前困惑、成长目标、价值倾向。"
        )

        top_chunks = self.retrieve(task=task)

        system_prompt = (
            f"你正在整理学生“{self.student_name}”的档案摘要。"
            "你的任务是生成一段简洁、清楚、适合给教师阅读的学生信息说明。"
            "必须依据提供的学生档案内容。"
            "不要编造。"
            "输出应是一段自然文字，不要分点，不要加标题。"
        )

        user_prompt = (
            f"以下是学生“{self.student_name}”的档案内容：\n\n"
            f"{self.build_context(top_chunks)}\n\n"
            "请整理成一段给教师使用的学生摘要。"
        )

        summary = self.answer_with_deepseek(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        if return_chunks:
            return {
                "student_name": self.student_name,
                "task": task,
                "profile_summary": summary,
                "chunks": top_chunks
            }
        return summary

    # =========================
    # 2.7 Prompt 构造
    # =========================
    def build_opening_question_system_prompt(self, hero_name: str) -> str:
        return (
            "你现在要扮演一个学生智能体。"
            f"这个学生的姓名是“{self.student_name}”。"
            "你必须严格依据提供的学生档案内容来组织表达。"
            "你的任务是替这个学生生成一段面向英雄人物的首轮提问。"
            "这段提问必须像真实学生会说的话，既要体现学生身份，也要体现他的专业、阶段、困惑和成长诉求。"
            f"提问对象是“{hero_name}”。"
            "表达要求自然、真诚、清楚，不要空泛，不要套话，不要变成老师总结。"
            "不要编造档案中没有的经历、家庭背景或具体事件。"
            "如果档案中没有某些细节，就不要强行补充。"
            "输出只需要一段完整提问，不要加分析，不要加标题，不要加引号。"
        )

    def build_opening_question_user_prompt(self, hero_name: str, top_chunks: List[Dict[str, Any]]) -> str:
        context = self.build_context(top_chunks)
        return (
            f"以下是学生“{self.student_name}”的档案内容：\n\n"
            f"{context}\n\n"
            f"现在请你以这名学生的身份，向英雄人物“{hero_name}”发出第一轮提问。"
            "要求提问中尽量自然包含：学生身份介绍、专业背景、当前困惑或关注问题，以及希望从这位英雄身上获得的启发。"
        )

    def build_student_speak_system_prompt(self) -> str:
        return (
            f"你是学生智能体“{self.student_name}”。"
            "你不是老师，也不是英雄人物。"
            "你必须依据学生档案内容，保持稳定的人设、表达风格、学习状态和成长诉求。"
            "你可以表达困惑、问题、反思、感想和阶段性理解。"
            "不要编造档案中没有明确支持的具体经历和细节。"
            "如果档案中没有足够信息，就保持克制自然，不要强行发挥。"
            "你的表达要像真实大学生，既不能过于幼稚，也不要变成老师式总结。"
        )

    def build_student_speak_user_prompt(self, task: str, top_chunks: List[Dict[str, Any]]) -> str:
        context = self.build_context(top_chunks)
        return (
            f"以下是学生“{self.student_name}”的档案内容：\n\n"
            f"{context}\n\n"
            f"现在请你以这名学生的身份完成下面这个任务：{task}\n\n"
            "请直接输出学生视角下的表达结果。"
        )

    # =========================
    # 2.8 生成能力
    # =========================
    def answer_with_deepseek(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        resp = self.deepseek_client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3
        )

        return resp.choices[0].message.content.strip()

    # =========================
    # 2.9 对外统一接口
    # =========================
    def retrieve(self, task: str) -> List[Dict[str, Any]]:
        vector_topk = self.search_top_k(
            task=task,
            top_k=self.recall_top_k
        )
        reranked_topn = self.rerank_with_jina(
            task=task,
            candidates=vector_topk,
            top_n=self.rerank_top_n
        )
        return reranked_topn

    def generate_opening_question(
        self,
        hero_name: str,
        return_chunks: bool = False
    ) -> Any:
        task = (
            f"为学生生成一段面向英雄人物“{hero_name}”的首轮提问，"
            "重点体现学生身份、专业、当前困惑和希望获得的启发。"
        )

        top_chunks = self.retrieve(task=task)

        system_prompt = self.build_opening_question_system_prompt(hero_name=hero_name)
        user_prompt = self.build_opening_question_user_prompt(
            hero_name=hero_name,
            top_chunks=top_chunks
        )
        question = self.answer_with_deepseek(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        if return_chunks:
            return {
                "student_name": self.student_name,
                "hero_name": hero_name,
                "task": task,
                "opening_question": question,
                "chunks": top_chunks
            }
        return question

    def speak(
        self,
        task: str,
        return_chunks: bool = False
    ) -> Any:
        top_chunks = self.retrieve(task=task)

        system_prompt = self.build_student_speak_system_prompt()
        user_prompt = self.build_student_speak_user_prompt(
            task=task,
            top_chunks=top_chunks
        )
        answer = self.answer_with_deepseek(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        if return_chunks:
            return {
                "student_name": self.student_name,
                "task": task,
                "answer": answer,
                "chunks": top_chunks
            }
        return answer


# =========================
# 3. 工具函数：路径与工厂
# =========================
def build_embedding_file(doc_id: str, embedding_root: Path = DEFAULT_EMBEDDING_ROOT) -> Path:
    return embedding_root / doc_id / f"{doc_id}_embeddings.jsonl"


def create_student_rag(
    student_name: str,
    embedding_file: Path,
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    use_proxy: bool = True,
) -> StudentRAGPipeline:
    return StudentRAGPipeline(
        embedding_file=embedding_file,
        student_name=student_name,
        recall_top_k=recall_top_k,
        rerank_top_n=rerank_top_n,
        use_proxy=use_proxy,
    )


def create_student_rag_by_doc_id(
    doc_id: str,
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    use_proxy: bool = True,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
) -> StudentRAGPipeline:
    embedding_file = build_embedding_file(doc_id=doc_id, embedding_root=embedding_root)
    return StudentRAGPipeline(
        embedding_file=embedding_file,
        recall_top_k=recall_top_k,
        rerank_top_n=rerank_top_n,
        use_proxy=use_proxy,
    )


# =========================
# 4. CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="学生智能体 Student RAG 脚本")

    parser.add_argument(
        "--doc_id",
        type=str,
        required=True,
        help="学生文档 ID，例如 student_wanghaoyu"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="opening",
        choices=["opening", "speak", "profile_summary"],
        help="运行模式：opening / speak / profile_summary，默认 opening"
    )
    parser.add_argument(
        "--hero_name",
        type=str,
        default=None,
        help="opening 模式下的目标英雄人物名称，例如 钱学森"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="speak 模式下的任务描述，例如 请你谈谈听完英雄回答后的感受"
    )
    parser.add_argument(
        "--recall_top_k",
        type=int,
        default=10,
        help="向量召回数量，默认 10"
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=5,
        help="rerank 保留数量，默认 5"
    )
    parser.add_argument(
        "--show_chunks",
        action="store_true",
        help="是否打印召回片段"
    )

    return parser


def run_cli():
    parser = build_arg_parser()
    args = parser.parse_args()

    rag = create_student_rag_by_doc_id(
        doc_id=args.doc_id,
        recall_top_k=args.recall_top_k,
        rerank_top_n=args.rerank_top_n,
    )

    print(f"已加载 {len(rag.records)} 条 embedding 记录")
    print(f"当前学生：{rag.student_name}")
    print(f"当前模式：{args.mode}")

    if args.mode == "profile_summary":
        result = rag.build_profile_summary(return_chunks=True)

        if args.show_chunks:
            print("\n===== 检索结果 Top-N =====")
            for i, ch in enumerate(result["chunks"], 1):
                preview = rag.normalize_text(ch.get("content"))[:120].replace("\n", " ")
                print(f"\n[{i}] vector_score={ch.get('score', 0):.4f}, rerank_score={ch.get('rerank_score', 0):.4f}")
                print(f"title={ch.get('title', '')}")
                print(f"chunk_type={ch.get('chunk_type', '')}")
                print(f"content={preview}...")

        print("\n===== 学生摘要 =====")
        print(result["profile_summary"])
        return

    if args.mode == "opening":
        if not args.hero_name:
            raise ValueError("opening 模式下必须传入 --hero_name")

        result = rag.generate_opening_question(
            hero_name=args.hero_name,
            return_chunks=True
        )

        if args.show_chunks:
            print("\n===== 检索结果 Top-N =====")
            for i, ch in enumerate(result["chunks"], 1):
                preview = rag.normalize_text(ch.get("content"))[:120].replace("\n", " ")
                print(f"\n[{i}] vector_score={ch.get('score', 0):.4f}, rerank_score={ch.get('rerank_score', 0):.4f}")
                print(f"title={ch.get('title', '')}")
                print(f"chunk_type={ch.get('chunk_type', '')}")
                print(f"content={preview}...")

        print("\n===== 生成的首轮提问 =====")
        print(result["opening_question"])
        return

    if args.mode == "speak":
        if not args.task:
            raise ValueError("speak 模式下必须传入 --task")

        result = rag.speak(
            task=args.task,
            return_chunks=True
        )

        if args.show_chunks:
            print("\n===== 检索结果 Top-N =====")
            for i, ch in enumerate(result["chunks"], 1):
                preview = rag.normalize_text(ch.get("content"))[:120].replace("\n", " ")
                print(f"\n[{i}] vector_score={ch.get('score', 0):.4f}, rerank_score={ch.get('rerank_score', 0):.4f}")
                print(f"title={ch.get('title', '')}")
                print(f"chunk_type={ch.get('chunk_type', '')}")
                print(f"content={preview}...")

        print("\n===== 学生表达结果 =====")
        print(result["answer"])
        return


if __name__ == "__main__":
    run_cli()