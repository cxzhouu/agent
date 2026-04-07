import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


class DeclarationStore:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.personal_dir = self.project_root / "data" / "declarations" / "personal"
        self.shared_dir = self.project_root / "data" / "declarations" / "shared"
        self.personal_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, base_dir: Path, doc_id: str) -> Path:
        return base_dir / f"{doc_id}.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load_personal(self, doc_id: str) -> Optional[Dict[str, Any]]:
        file_path = self._file(self.personal_dir, doc_id)
        if not file_path.exists():
            return None
        return json.loads(file_path.read_text(encoding="utf-8"))

    def save_declaration(
        self,
        doc_id: str,
        student_name: str,
        mission: str,
        keep_path: str,
        role_model: str,
        teacher_comment: str = "",
    ) -> Dict[str, Any]:
        data = {
            "doc_id": doc_id,
            "student_name": student_name,
            "mission": mission.strip(),
            "keep_path": keep_path.strip(),
            "role_model": role_model.strip(),
            "teacher_comment": teacher_comment.strip(),
            "updated_at": self._now_iso(),
        }

        personal_file = self._file(self.personal_dir, doc_id)
        shared_file = self._file(self.shared_dir, doc_id)
        payload = json.dumps(data, ensure_ascii=False, indent=2)

        personal_file.write_text(payload, encoding="utf-8")
        shared_file.write_text(payload, encoding="utf-8")
        return data

    def update_comment(self, doc_id: str, teacher_comment: str) -> Dict[str, Any]:
        current = self.load_personal(doc_id)
        if not current:
            raise FileNotFoundError(f"未找到该学生的成人宣言：{doc_id}")

        return self.save_declaration(
            doc_id=doc_id,
            student_name=current.get("student_name", doc_id),
            mission=current.get("mission", ""),
            keep_path=current.get("keep_path", ""),
            role_model=current.get("role_model", ""),
            teacher_comment=teacher_comment,
        )

    def list_shared(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for file_path in sorted(self.shared_dir.glob("*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            rows.append(data)
        return rows
