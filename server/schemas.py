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
    markdown are also written to output_dir on disk regardless (see
    DocumentPipeline.process_pdf_group/process_image_group), but are
    included here too so callers don't have to read them back off disk."""

    file: str
    status: str  # "success" | "error"
    output_dir: str | None = None
    json_data: dict | None = Field(default=None, alias="json")
    markdown: str | None = None
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class BatchOCRResponse(BaseModel):
    results: list[BatchItemResult]
