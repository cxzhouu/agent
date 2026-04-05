from typing import List, Dict


class HeroService:
    HEROES = {
        "li_dazhao": {"doc_id": "li_dazhao", "name": "李大钊"},
        "jiao_yulu": {"doc_id": "jiao_yulu", "name": "焦裕禄"},
        "qian_xuesen": {"doc_id": "qian_xuesen", "name": "钱学森"},
        "huang_danian": {"doc_id": "huang_danian", "name": "黄大年"},
        "huang_wenxiu": {"doc_id": "huang_wenxiu", "name": "黄文秀"},
    }

    def list_heroes(self) -> List[Dict]:
        return list(self.HEROES.values())

    def get_hero(self, doc_id: str) -> Dict:
        if doc_id not in self.HEROES:
            raise ValueError(f"未支持的英雄：{doc_id}")
        return self.HEROES[doc_id]