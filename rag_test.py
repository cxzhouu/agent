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

DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "data" / "embedding"

DEFAULT_HEROES = {
    "li_dazhao": "李大钊",
    "jiao_yulu": "焦裕禄",
    "qian_xuesen": "钱学森",
    "huang_danian": "黄大年",
    "huang_wenxiu": "黄文秀",
}

SUPPORTED_AGENT_ROLES = {"hero", "teacher"}


# =========================
# 2. 通用 Hero / Teacher RAG Pipeline
# =========================
class HeroRAGPipeline:
    def __init__(
        self,
        embedding_file: Path,
        hero_name: str,
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
        self.hero_name = hero_name

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
    def validate_agent_role(agent_role: str) -> str:
        role = (agent_role or "hero").strip().lower()
        if role not in SUPPORTED_AGENT_ROLES:
            raise ValueError(
                f"不支持的 agent_role: {agent_role}，当前支持: {', '.join(sorted(SUPPORTED_AGENT_ROLES))}"
            )
        return role

    # =========================
    # 2.2 查询构造
    # =========================
    def build_query_text(self, question: str, agent_role: str = "hero") -> str:
        """
        仅用于英雄模式检索。
        老师模式不走这里，老师必须走 teacher_summary。
        """
        role = self.validate_agent_role(agent_role)

        if role != "hero":
            raise ValueError("老师模式不允许使用 build_query_text，请改用 teacher_summary()")

        return (
            f"人物：{self.hero_name}\n"
            "场景：学生智能体向英雄人物提问\n"
            f"内容：{question.strip()}"
        )

    def build_teacher_summary_query_text(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str
    ) -> str:
        return (
            f"人物：{self.hero_name}\n"
            "场景：教师需要基于英雄人物知识，对学生提问和英雄回答进行课堂总结引导\n"
            f"学生信息：{student_profile.strip()}\n"
            f"学生提问：{student_question.strip()}\n"
            f"英雄回答：{hero_answer.strip()}"
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
        question: str,
        agent_role: str = "hero",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        role = self.validate_agent_role(agent_role)
        if role != "hero":
            raise ValueError("老师模式不允许使用 search_top_k，请改用 teacher_summary()")

        top_k = top_k or self.recall_top_k
        query_text = self.build_query_text(question, agent_role=role)
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
                "entity_name": rec.get("entity_name"),
                "tags": rec.get("tags"),
                "question": rec.get("question"),
                "answer": rec.get("answer"),
                "source_file": rec.get("source_file"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_teacher_summary_top_k(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.recall_top_k
        query_text = self.build_teacher_summary_query_text(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer,
        )
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
        entity_name = self.normalize_text(candidate.get("entity_name"))
        title = self.normalize_text(candidate.get("title"))
        chunk_type = self.normalize_text(candidate.get("chunk_type"))
        section_path = self.normalize_text(candidate.get("section_path"))
        tags = self.normalize_text(candidate.get("tags"))
        content = self.normalize_text(candidate.get("content"))
        question = self.normalize_text(candidate.get("question"))
        answer = self.normalize_text(candidate.get("answer"))

        lines = []

        if entity_name:
            lines.append(f"人物名称：{entity_name}")
        if title:
            lines.append(f"标题：{title}")
        if chunk_type:
            lines.append(f"知识类型：{chunk_type}")
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
        question: str,
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
            "query": question,
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
    # 2.5 构造上下文
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
                lines.append(f"知识内容：{content}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # =========================
    # 2.6 提示词构造
    # =========================
    def build_system_prompt(self, agent_role: str = "hero") -> str:
        role = self.validate_agent_role(agent_role)

        if role == "teacher":
            return (
                "你是一名思政课教师。"
                "你不是英雄人物本人，不要使用第一人称扮演英雄。"
                "你的任务不是重复英雄的话，也不是脱离情境空泛说教。"
                "你需要综合学生的个人背景与困惑、学生提出的问题、英雄刚才给出的回答，以及英雄人物相关知识，做出课堂中的最后总结与引导。"
                "你的输出应完成三件事："
                "第一，先概括英雄回答的核心意思；"
                "第二，再结合学生的专业、年级、当前困惑，说明这段回答对这个学生意味着什么；"
                "第三，给出一段有方向感、可执行、不过度空泛的成长建议。"
                "表达要清楚、自然、有层次，适合课堂总结。"
                "请尽量控制在180字以内，避免过长。"
                "只能依据提供的信息和知识内容作答，不得编造资料中没有明确支持的事实、经历、情绪、细节或对话。"
                "如果知识中没有足够依据，请直接说“现有资料未明确记载”或“从现有材料看，这一点还不能下过于确定的判断”。"
                "不要出现“我正在检索资料”“根据知识库”“结合检索结果”等暴露系统过程的话。"
            )

        return (
            f"你是{self.hero_name}本人，当前正在与学生智能体直接对话。"
            "你必须始终使用第一人称“我”来回答。"
            "你回答时要像真实人物在交流，而不是像助手在总结资料。"
            "你的回答要适合课堂思政对话场景，既要有人物感，也要有教育引导意义。"
            "你只能依据提供的知识内容回答，不得编造资料中没有明确支持的事实、经历、情绪、细节或对话。"
            "如果知识中没有足够依据，请直接说“这件事按我现有掌握的内容，还不能说得过于确定”或“现有资料未明确记载”。"
            "绝对不要出现“根据资料”“从资料中看”“依据资料1”“结合上述内容”“文档显示”“检索结果表明”这类说法。"
            "绝对不要暴露你正在参考资料、知识库、检索结果或上下文。"
            "表达要自然、庄重、清晰，符合人物身份与时代气质。"
            "面对学生关于初心、成长、成才、成功、选择、奉献、基层实践等问题时，可以在已有知识支持范围内给出有方向感的回应。"
            "先直接回答问题，再作展开。"
            "请尽量控制在180字以内，避免过长。"
        )

    def build_user_prompt(
        self,
        question: str,
        top_chunks: List[Dict[str, Any]],
        agent_role: str = "hero"
    ) -> str:
        role = self.validate_agent_role(agent_role)
        if role != "hero":
            raise ValueError("老师模式不允许使用 build_user_prompt，请改用 teacher_summary()")

        context = self.build_context(top_chunks)

        return (
            f"以下是你所掌握的与自己相关的知识内容：\n\n"
            f"{context}\n\n"
            f"学生智能体现在问你：{question}\n\n"
            f"请直接以{self.hero_name}本人的身份回答。"
        )

    def build_teacher_summary_user_prompt(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str,
        top_chunks: List[Dict[str, Any]]
    ) -> str:
        context = self.build_context(top_chunks)
        return (
            f"以下是与英雄人物“{self.hero_name}”相关的知识内容：\n\n"
            f"{context}\n\n"
            f"学生信息：{student_profile}\n\n"
            f"学生提问：{student_question}\n\n"
            f"英雄回答：{hero_answer}\n\n"
            "请你以教师身份，做最后的课堂总结与引导。"
        )

    # =========================
    # 2.7 生成回答
    # =========================
    def answer_with_deepseek(
        self,
        question: str,
        top_chunks: List[Dict[str, Any]],
        agent_role: str = "hero"
    ) -> str:
        role = self.validate_agent_role(agent_role)
        if role != "hero":
            raise ValueError("老师模式不允许使用 answer_with_deepseek，请改用 teacher_summary()")

        system_prompt = self.build_system_prompt(agent_role=role)
        user_prompt = self.build_user_prompt(
            question=question,
            top_chunks=top_chunks,
            agent_role=role
        )

        resp = self.deepseek_client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )

        return resp.choices[0].message.content

    def answer_teacher_summary_with_deepseek(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str,
        top_chunks: List[Dict[str, Any]]
    ) -> str:
        system_prompt = self.build_system_prompt(agent_role="teacher")
        user_prompt = self.build_teacher_summary_user_prompt(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer,
            top_chunks=top_chunks
        )

        resp = self.deepseek_client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )

        return resp.choices[0].message.content

    # =========================
    # 2.8 对外统一调用接口
    # =========================
    def retrieve(
        self,
        question: str,
        agent_role: str = "hero"
    ) -> List[Dict[str, Any]]:
        role = self.validate_agent_role(agent_role)
        if role != "hero":
            raise ValueError("老师模式不允许使用 retrieve(question, agent_role='teacher')，请改用 teacher_summary()")

        vector_topk = self.search_top_k(
            question=question,
            agent_role=role,
            top_k=self.recall_top_k
        )
        reranked_topn = self.rerank_with_jina(
            question=question,
            candidates=vector_topk,
            top_n=self.rerank_top_n
        )
        return reranked_topn

    def retrieve_teacher_summary(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str
    ) -> List[Dict[str, Any]]:
        query_text = self.build_teacher_summary_query_text(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer
        )

        vector_topk = self.search_teacher_summary_top_k(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer,
            top_k=self.recall_top_k
        )
        reranked_topn = self.rerank_with_jina(
            question=query_text,
            candidates=vector_topk,
            top_n=self.rerank_top_n
        )
        return reranked_topn

    def ask(
        self,
        question: str,
        agent_role: str = "hero",
        return_chunks: bool = False
    ) -> Any:
        role = self.validate_agent_role(agent_role)

        if role != "hero":
            raise ValueError("老师模式已禁用 ask(..., agent_role='teacher')，请改用 teacher_summary()")

        top_chunks = self.retrieve(
            question=question,
            agent_role=role
        )
        answer = self.answer_with_deepseek(
            question=question,
            top_chunks=top_chunks,
            agent_role=role
        )

        if return_chunks:
            return {
                "hero_name": self.hero_name,
                "agent_role": role,
                "question": question,
                "answer": answer,
                "chunks": top_chunks
            }
        return answer

    def teacher_summary(
        self,
        student_profile: str,
        student_question: str,
        hero_answer: str,
        return_chunks: bool = False
    ) -> Any:
        top_chunks = self.retrieve_teacher_summary(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer
        )

        answer = self.answer_teacher_summary_with_deepseek(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer,
            top_chunks=top_chunks
        )

        if return_chunks:
            return {
                "hero_name": self.hero_name,
                "agent_role": "teacher",
                "student_profile": student_profile,
                "student_question": student_question,
                "hero_answer": hero_answer,
                "answer": answer,
                "chunks": top_chunks
            }
        return answer


# =========================
# 3. 工具函数：路径与工厂
# =========================
def build_embedding_file(doc_id: str, embedding_root: Path = DEFAULT_EMBEDDING_ROOT) -> Path:
    return embedding_root / doc_id / f"{doc_id}_embeddings.jsonl"


def create_hero_rag(
    hero_name: str,
    embedding_file: Path,
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    use_proxy: bool = True,
) -> HeroRAGPipeline:
    return HeroRAGPipeline(
        embedding_file=embedding_file,
        hero_name=hero_name,
        recall_top_k=recall_top_k,
        rerank_top_n=rerank_top_n,
        use_proxy=use_proxy,
    )


def create_hero_rag_by_doc_id(
    doc_id: str,
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    use_proxy: bool = True,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
) -> HeroRAGPipeline:
    if doc_id not in DEFAULT_HEROES:
        raise ValueError(
            f"未支持的 doc_id: {doc_id}，当前支持: {', '.join(DEFAULT_HEROES.keys())}"
        )

    hero_name = DEFAULT_HEROES[doc_id]
    embedding_file = build_embedding_file(doc_id=doc_id, embedding_root=embedding_root)

    return HeroRAGPipeline(
        embedding_file=embedding_file,
        hero_name=hero_name,
        recall_top_k=recall_top_k,
        rerank_top_n=rerank_top_n,
        use_proxy=use_proxy,
    )


def create_all_hero_rags(
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    use_proxy: bool = True,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
) -> Dict[str, HeroRAGPipeline]:
    pipelines = {}
    for doc_id, hero_name in DEFAULT_HEROES.items():
        embedding_file = build_embedding_file(doc_id=doc_id, embedding_root=embedding_root)
        pipelines[doc_id] = HeroRAGPipeline(
            embedding_file=embedding_file,
            hero_name=hero_name,
            recall_top_k=recall_top_k,
            rerank_top_n=rerank_top_n,
            use_proxy=use_proxy,
        )
    return pipelines


# =========================
# 4. 命令行测试入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用英雄 / 教师总结双模式 RAG 脚本")

    parser.add_argument(
        "--doc_id",
        type=str,
        required=True,
        help="人物文档 ID，例如 li_dazhao / jiao_yulu / qian_xuesen"
    )
    parser.add_argument(
        "--agent_role",
        type=str,
        default="hero",
        choices=["hero", "teacher"],
        help="回答角色：hero 或 teacher，默认 hero"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="hero 模式下直接传入问题"
    )
    parser.add_argument(
        "--student_profile",
        type=str,
        default=None,
        help="teacher 模式下学生信息摘要"
    )
    parser.add_argument(
        "--student_question",
        type=str,
        default=None,
        help="teacher 模式下学生提问"
    )
    parser.add_argument(
        "--hero_answer",
        type=str,
        default=None,
        help="teacher 模式下英雄回答"
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

    rag = create_hero_rag_by_doc_id(
        doc_id=args.doc_id,
        recall_top_k=args.recall_top_k,
        rerank_top_n=args.rerank_top_n,
    )

    print(f"已加载 {len(rag.records)} 条 embedding 记录")
    print(f"当前人物：{rag.hero_name}")
    print(f"当前模式：{args.agent_role}")

    if args.agent_role == "teacher":
        if not args.student_profile or not args.student_question or not args.hero_answer:
            raise ValueError("teacher 模式下必须同时传入 --student_profile --student_question --hero_answer")

        result = rag.teacher_summary(
            student_profile=args.student_profile,
            student_question=args.student_question,
            hero_answer=args.hero_answer,
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

        print("\n===== 教师总结 =====")
        print(result["answer"])
        return

    if args.question:
        result = rag.ask(
            args.question,
            agent_role="hero",
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

        print("\n===== 模型回答 =====")
        print(result["answer"])
        return

    while True:
        question = input("请输入问题（输入 exit 退出）：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = rag.ask(
            question,
            agent_role="hero",
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

        print("\n===== 模型回答 =====")
        print(result["answer"])


if __name__ == "__main__":
    run_cli()