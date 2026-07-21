from pydantic import BaseModel, ConfigDict, Field


class OCRResponse(BaseModel):
    file: str
    json_data: dict = Field(alias="json")
    markdown: str

    model_config = ConfigDict(populate_by_name=True)


class ErrorResponse(BaseModel):
    detail: str
