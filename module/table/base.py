from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class TableProcessor(Protocol):
    """Contract a swappable table-extraction backend must satisfy: a
    deskewed table crop in, block 'content' string out (Markdown or HTML --
    build_markdown/build_json splice it in as-is, no format-specific
    parsing)."""

    def __call__(
        self,
        crop: Image.Image,
        debug_dir: str | None = None,
        pn: int = 0,
        tno: int = 0,
    ) -> str: ...
