from typing import List, Optional
from pydantic import BaseModel, Field


class CreateStudentRequest(BaseModel):
    name: str = Field(..., description="学生姓名")
    grade: str = Field(..., description="年级")
    major: str = Field(..., description="专业")

    identity: Optional[str] = Field(default="", description="当前身份描述")
    self_view: Optional[str] = Field(default="", description="自我认知")

    personality_traits: List[str] = Field(default_factory=list, description="性格特点")
    speaking_style: List[str] = Field(default_factory=list, description="表达风格")

    interests: List[str] = Field(default_factory=list, description="兴趣方向")
    strengths: List[str] = Field(default_factory=list, description="当前擅长")
    weaknesses: List[str] = Field(default_factory=list, description="当前薄弱点")

    current_stage: Optional[str] = Field(default="", description="当前学习阶段")
    learning_goals: List[str] = Field(default_factory=list, description="学习目标")
    current_confusions: List[str] = Field(default_factory=list, description="当前困惑")

    concerns: List[str] = Field(default_factory=list, description="关注的问题")
    values: List[str] = Field(default_factory=list, description="认可的价值观")
    expectations: Optional[str] = Field(default="", description="对未来的期待")

    common_questions: List[str] = Field(default_factory=list, description="常见提问方式")
    response_preference: List[str] = Field(default_factory=list, description="适合的回应方式")

    opening_intro: Optional[str] = Field(default="", description="面向英雄提问时的身份介绍")
    opening_problem: Optional[str] = Field(default="", description="当前最想解决的问题")
    opening_expectation: Optional[str] = Field(default="", description="希望从英雄身上得到的启发")


class CreateSingleSessionRequest(BaseModel):
    student_doc_id: str
    hero_doc_id: str


class CreateMultiSessionRequest(BaseModel):
    student_doc_id: str
    hero_doc_ids: List[str]


class NextRoundRequest(BaseModel):
    session_id: str