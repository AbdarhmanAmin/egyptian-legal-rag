from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
  question: str = Field(
      ...,
      min_length=1,
      description="السؤال القانوني الموجه للنظام حول القانون المدني المصري",
  )

  @field_validator("question")
  @classmethod
  def check_not_empty(cls, v: str) -> str:
    # الـ Validation لمنع النُصوص الفراغية (مثل "   ") وإرجاع 422 تلقائياً
    if not v.strip():
      raise ValueError("لا يمكن أن يكون السؤال فارغاً أو يحتوي على مسافات فقط.")
    return v.strip()


class AskResponse(BaseModel):
  answer: str
  sources: list[str]


class HealthResponse(BaseModel):
  status: str = "healthy"
  documents_indexed: int