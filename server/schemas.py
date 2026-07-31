from pydantic import BaseModel, ConfigDict, Field


class OCRResponse(BaseModel):
    file: str
    json_data: dict = Field(alias="json")
    markdown: str

    model_config = ConfigDict(populate_by_name=True)


class ErrorResponse(BaseModel):
    detail: str


class BatchItemResult(BaseModel):
    """One file's outcome within a /pdfs or /images batch request. json/
    markdown are NOT included here (unlike OCRResponse) -- they're already
    written to output_dir on disk by the time this is returned, and a batch
    can contain hundreds of files, so embedding full content per item risks
    a huge/slow response."""

    file: str
    status: str  # "success" | "error"
    output_dir: str | None = None
    error: str | None = None


class BatchOCRResponse(BaseModel):
    results: list[BatchItemResult]
