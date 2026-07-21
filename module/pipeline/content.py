import re

from ..layout import LayoutLabelSchema
from .types import PageBlock

# ── Ghép nội dung block: line break <br> + tự nhận diện list item ──────────
#
# Nhận diện 1 dòng là "list item" nếu nó bắt đầu bằng marker kiểu:
#   a) b) c)  /  a. b. c.  /  1) 2) 3)  /  1. 2. 3.
# (chữ cái đơn hoặc số, theo sau bởi ')' hoặc '.', rồi đến khoảng trắng)
ALPHA_ITEM_RE = re.compile(r"^\s*[a-zđ][\.\)]\s+")  # a) b) c) đ)...
# CHỈ chữ THƯỜNG -- nếu cho phép cả chữ HOA, tiêu đề mục La Mã kiểu
# 'I. Người sử dụng đất...' (1 ký tự HOA + dấu chấm) sẽ bị nhận NHẦM thành
# marker list 'alpha', kéo theo mọi dòng thường phía sau bị fix
# nối-dòng-word-wrap nuốt chung thành 1 khối, mất hết <br> phân dòng.
NUM_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s+")  # 1. 2. 3)...
MIN_LIST_ITEMS = 2
# cần ÍT NHẤT bấy nhiêu dòng list-item LIÊN TIẾP, CÙNG NHÓM (cùng là chữ
# cái, hoặc cùng là số) trong CÙNG 1 block mới được bọc <ul><li>. Marker
# khác nhóm (số xen chữ cái) hoặc chỉ có 1 dòng lẻ -> giữ nguyên text
# thường, KHÔNG bọc <ul> -- tránh hiện bullet giả khiến nhiều dòng/block
# không liên quan trông như 1 danh sách liên tục.


def _list_item_type(line: str) -> str | None:
    """Trả về 'alpha' nếu khớp marker chữ cái (a) b) c)...), 'num' nếu khớp
    marker số (1. 2. 3)...), None nếu không phải list item."""
    if NUM_ITEM_RE.match(line):
        return "num"
    if ALPHA_ITEM_RE.match(line):
        return "alpha"
    return None


def build_block_content(line_texts: list[str]) -> str:
    """
    Ghép các "hàng" (mỗi hàng = 1 dòng chữ thật, đã gom theo tb_rows) thành
    1 chuỗi content dạng HTML-trong-Markdown:
      - Các dòng THƯỜNG liên tiếp -> nối bằng '<br>\n' (giữ xuống dòng khi
        render Markdown, vì '\n' đơn thuần bị coi là khoảng trắng).
      - Các dòng LIST ITEM liên tiếp, CÙNG NHÓM marker (toàn 'alpha' hoặc
        toàn 'num') -> gom thành 1 khối <ul>, mỗi dòng là 1 <li>...</li> --
        chỉ khi số dòng liên tiếp cùng nhóm >= MIN_LIST_ITEMS.
      - Nếu marker đổi nhóm giữa chừng (ví dụ '1. ...' rồi 'a) ...') ->
        đóng khối list hiện tại (theo đúng rule MIN_LIST_ITEMS ở trên) và
        mở khối MỚI cho nhóm marker khác, KHÔNG gộp chung 1 <ul>.
      - Nếu chỉ có 1 dòng lẻ khớp marker (dù đứng riêng hay bị đổi nhóm
        ngay sau đó), coi như text thường, không bọc <ul>.
      - Các khối (text-run hoặc <ul>) nối với nhau bằng '\n'.
    """
    segments = []  # list các đoạn HTML đã hoàn chỉnh (text-run hoặc <ul>)
    buffer = []  # dòng thường đang gom (chưa flush)
    list_buffer = []  # dòng list-item đang gom (chưa flush)
    list_type = None  # 'alpha' hoặc 'num' -- nhóm marker của list_buffer hiện tại

    def flush_buffer():
        if buffer:
            segments.append(("text", "<br>\n".join(buffer)))
            buffer.clear()

    def flush_list():
        nonlocal list_type
        if len(list_buffer) >= MIN_LIST_ITEMS:
            items = "\n".join(f"<li>{t}</li>" for t in list_buffer)
            segments.append(("ul", f"<ul>\n{items}\n</ul>"))
        elif len(list_buffer) == 1:
            segments.append(("text", list_buffer[0]))
        list_buffer.clear()
        list_type = None

    for line in line_texts:
        t = _list_item_type(line)
        if t is not None:
            if list_buffer and t != list_type:
                # marker đổi nhóm (num <-> alpha) -- đóng khối cũ, mở khối mới
                flush_list()
            if not list_buffer:
                flush_buffer()  # chuyển từ text-mode sang list-mode
            list_buffer.append(line)
            list_type = t
        else:
            if list_buffer:
                # dòng không có marker nhưng đang ở GIỮA 1 danh sách
                # (list_buffer chưa flush) -- coi đây là phần bị NGẮT XUỐNG
                # DÒNG (word-wrap) của CHÍNH item cuối cùng, nối tiếp vào
                # item đó thay vì bẻ gãy danh sách.
                list_buffer[-1] = list_buffer[-1] + " " + line
            else:
                buffer.append(line)
    flush_buffer()
    flush_list()

    if not segments:
        return ""

    result = segments[0][1]
    for i in range(1, len(segments)):
        prev_kind, _ = segments[i - 1]
        cur_kind, cur_content = segments[i]
        sep = "\n" if (prev_kind == "ul" or cur_kind == "ul") else "<br>\n"
        result += sep + cur_content
    return result


# ── JSON / Markdown builders ──────────────────────────────────────────────────


def build_json(file_label: str, pages_blocks: list[list[PageBlock]]) -> dict:
    return {
        "file": file_label,
        "pages": [
            {
                "page": pn + 1,
                "blocks": [
                    {
                        "id": j + 1,
                        "type": b["type"],
                        "bbox": [round(v, 1) for v in b["bbox"]],
                        "score": round(b.get("score", 0), 4),
                        "content_type": b["content_type"],
                        "content": b.get("content"),
                    }
                    for j, b in enumerate(blocks)
                ],
            }
            for pn, blocks in enumerate(pages_blocks)
        ],
    }


def build_markdown(pages_blocks: list[list[PageBlock]], label_schema: LayoutLabelSchema) -> str:
    """
    Mỗi block là 1 khối Markdown riêng (heading / paragraph / table) nên
    PHẢI nối bằng dòng trống ('\n\n'), không phải 1 '\n' -- nếu chỉ nối
    bằng 1 '\n', nhiều trình render Markdown sẽ gộp 2 block liền kề thành
    cùng 1 dòng hiển thị.
    """
    lines = []
    for pn, blocks in enumerate(pages_blocks):
        lines.append(f"<!-- Page {pn + 1} -->")
        for b in blocks:
            if b["content_type"] == "skip":
                continue
            content = (b.get("content") or "").strip()
            if not content:
                continue
            btype = b["type"].lower()
            if b["content_type"] == "table":
                lines.append(content)
            elif btype in label_schema.title_types:
                lines.append(f"# {content}")
            elif btype in label_schema.h2_types:
                lines.append(f"## {content}")
            else:
                lines.append(content)
    return "\n\n".join(lines)
