from pathlib import Path
from typing import Dict, Any, List

from rag_student import create_student_rag_by_doc_id
from rag_test import create_hero_rag_by_doc_id

from services.session_store import SessionStore
from services.hero_service import HeroService


class DialogueService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.session_store = SessionStore(project_root)
        self.hero_service = HeroService()

    def create_single_session(self, student_doc_id: str, hero_doc_id: str) -> Dict[str, Any]:
        payload = {
            "student_doc_id": student_doc_id,
            "hero_doc_id": hero_doc_id,
        }
        return self.session_store.create_session("single", payload)

    def create_multi_session(self, student_doc_id: str, hero_doc_ids: List[str]) -> Dict[str, Any]:
        if not (2 <= len(hero_doc_ids) <= 5):
            raise ValueError("多英雄模式下，英雄数量必须在 2 到 5 之间。")

        payload = {
            "student_doc_id": student_doc_id,
            "hero_doc_ids": hero_doc_ids,
        }
        return self.session_store.create_session("multi", payload)

    def run_next_round(self, session_id: str) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        session_type = session["session_type"]

        if session_type == "single":
            return self._run_single_round(session)

        if session_type == "multi":
            return self._run_multi_round(session)

        raise ValueError(f"未知会话类型：{session_type}")

    def _run_single_round(self, session: Dict[str, Any]) -> Dict[str, Any]:
        student_doc_id = session["payload"]["student_doc_id"]
        hero_doc_id = session["payload"]["hero_doc_id"]

        student_rag = create_student_rag_by_doc_id(student_doc_id)
        hero_rag = create_hero_rag_by_doc_id(hero_doc_id)
        teacher_rag = create_hero_rag_by_doc_id(hero_doc_id)

        hero_meta = self.hero_service.get_hero(hero_doc_id)
        hero_name = hero_meta["name"]

        student_question = student_rag.generate_opening_question(hero_name=hero_name)
        hero_answer = hero_rag.ask(student_question, agent_role="hero")
        student_profile = student_rag.build_profile_summary()
        teacher_summary = teacher_rag.teacher_summary(
            student_profile=student_profile,
            student_question=student_question,
            hero_answer=hero_answer
        )

        round_data = {
            "student_question": student_question,
            "hero_doc_id": hero_doc_id,
            "hero_name": hero_name,
            "hero_answer": hero_answer,
            "teacher_summary": teacher_summary,
        }

        self.session_store.append_round(session["session_id"], round_data)
        return round_data

    def _run_multi_round(self, session: Dict[str, Any]) -> Dict[str, Any]:
        student_doc_id = session["payload"]["student_doc_id"]
        hero_doc_ids = session["payload"]["hero_doc_ids"]

        student_rag = create_student_rag_by_doc_id(student_doc_id)
        student_profile = student_rag.build_profile_summary()

        hero_name_list = [self.hero_service.get_hero(x)["name"] for x in hero_doc_ids]
        hero_names_text = "、".join(hero_name_list)

        student_question = student_rag.generate_opening_question(hero_name=hero_names_text)

        hero_rounds = []
        for hero_doc_id in hero_doc_ids:
            hero_meta = self.hero_service.get_hero(hero_doc_id)
            hero_name = hero_meta["name"]

            hero_rag = create_hero_rag_by_doc_id(hero_doc_id)
            teacher_rag = create_hero_rag_by_doc_id(hero_doc_id)

            hero_answer = hero_rag.ask(student_question, agent_role="hero")
            teacher_summary = teacher_rag.teacher_summary(
                student_profile=student_profile,
                student_question=student_question,
                hero_answer=hero_answer
            )

            hero_rounds.append({
                "hero_doc_id": hero_doc_id,
                "hero_name": hero_name,
                "hero_answer": hero_answer,
                "teacher_summary": teacher_summary,
            })

        round_data = {
            "student_question": student_question,
            "hero_rounds": hero_rounds,
        }

        self.session_store.append_round(session["session_id"], round_data)
        return round_data