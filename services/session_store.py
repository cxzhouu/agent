import json
import uuid
from pathlib import Path
from typing import Dict, Any, List


class SessionStore:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.store_dir = self.project_root / "data" / "sessions"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, session_id: str) -> Path:
        return self.store_dir / f"{session_id}.json"

    def create_session(self, session_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        data = {
            "session_id": session_id,
            "session_type": session_type,
            "payload": payload,
            "rounds": [],
        }
        self._file(session_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return data

    def load_session(self, session_id: str) -> Dict[str, Any]:
        file_path = self._file(session_id)
        if not file_path.exists():
            raise FileNotFoundError(f"未找到会话：{session_id}")
        return json.loads(file_path.read_text(encoding="utf-8"))

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        self._file(session_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def append_round(self, session_id: str, round_data: Dict[str, Any]) -> Dict[str, Any]:
        data = self.load_session(session_id)
        data["rounds"].append(round_data)
        self.save_session(session_id, data)
        return data

    def list_sessions(self) -> List[Dict[str, Any]]:
        results = []
        for file_path in sorted(self.store_dir.glob("*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            results.append({
                "session_id": data["session_id"],
                "session_type": data["session_type"],
                "round_count": len(data.get("rounds", [])),
            })
        return results