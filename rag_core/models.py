from dataclasses import dataclass


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chunk:
    id: str
    text: str
    page_number: int
