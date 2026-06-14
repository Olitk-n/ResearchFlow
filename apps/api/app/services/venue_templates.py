import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..config import get_settings

OFFICIAL_TEMPLATES = {
    "iclr": {
        "year": 2026,
        "url": "https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip",
        "host": "github.com",
        "style": "iclr2026_conference.sty",
        "guide": "https://iclr.cc/Conferences/2026/AuthorGuide",
    },
    "icml": {
        "year": 2026,
        "url": "https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip",
        "host": "media.icml.cc",
        "style": "icml2026.sty",
        "guide": "https://icml.cc/Conferences/2026/AuthorInstructions",
    },
    "neurips": {
        "year": 2026,
        "url": (
            "https://media.neurips.cc/Conferences/NeurIPS2026/"
            "Formatting_Instructions_For_NeurIPS_2026.zip"
        ),
        "host": "media.neurips.cc",
        "style": "neurips_2026.sty",
        "guide": "https://neurips.cc/Conferences/2026/CallForPapers",
    },
}

COPY_SUFFIXES = {
    ".sty",
    ".bst",
    ".cls",
    ".bbx",
    ".cbx",
    ".def",
    ".cfg",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}


class TemplateUnavailable(RuntimeError):
    pass


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise TemplateUnavailable("official template archive contains an unsafe path")
        bundle.extractall(destination)


def _download(url: str, destination: Path, expected_host: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != expected_host:
        raise TemplateUnavailable("template source is not an approved official HTTPS host")
    request = Request(url, headers={"User-Agent": "ResearchFlow/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read(30 * 1024 * 1024 + 1)
    except Exception as exc:
        raise TemplateUnavailable(
            "无法取得官方投稿模板；请联网后重试，系统不会用通用 article 冒充会议模板"
        ) from exc
    if len(payload) > 30 * 1024 * 1024:
        raise TemplateUnavailable("official template archive exceeded the 30 MB limit")
    destination.write_bytes(payload)


def ensure_official_template(target: str, manuscript_root: Path) -> dict:
    if target == "arxiv":
        return {
            "target": "arxiv",
            "source": "built-in article scaffold",
            "official_template": False,
        }
    publisher_scaffolds = {
        "ieee_conference": {
            "publisher": "IEEE",
            "class_file": "IEEEtran",
            "source_url": "https://template-selector.ieee.org/",
            "author_guide": (
                "https://conferences.ieeeauthorcenter.ieee.org/"
                "write-your-paper/authoring-tools-and-templates/"
            ),
        },
        "elsevier_journal": {
            "publisher": "Elsevier",
            "class_file": "elsarticle",
            "source_url": (
                "https://www.elsevier.com/researcher/author/"
                "policies-and-guidelines/latex-instructions"
            ),
            "author_guide": (
                "https://service.elsevier.com/app/answers/detail/a_id/5955/"
                "supporthub/publishing/"
            ),
        },
    }
    if target in publisher_scaffolds:
        profile = publisher_scaffolds[target]
        metadata = {
            "target": target,
            **profile,
            "official_template": False,
            "publisher_scaffold": True,
            "requires_specific_publication_check": True,
            "prepared_at": datetime.now(UTC).isoformat(),
        }
        (manuscript_root / "venue-template.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata
    profile = OFFICIAL_TEMPLATES[target]
    cache = get_settings().storage_root / "venue-templates" / f"{target}-{profile['year']}"
    archive = cache / "official-template.zip"
    extracted = cache / "extracted"
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        _download(profile["url"], archive, profile["host"])
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not extracted.exists():
        extracted.mkdir()
        try:
            _safe_extract(archive, extracted)
        except Exception:
            shutil.rmtree(extracted, ignore_errors=True)
            raise
    expected = list(extracted.rglob(profile["style"]))
    if not expected:
        raise TemplateUnavailable(
            f"官方模板中未找到预期样式文件 {profile['style']}"
        )
    copied = []
    for source in extracted.rglob("*"):
        if not source.is_file() or source.suffix.casefold() not in COPY_SUFFIXES:
            continue
        destination = manuscript_root / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise TemplateUnavailable(f"官方模板包含重名且内容不同的文件：{source.name}")
        shutil.copy2(source, destination)
        copied.append(source.name)
    metadata = {
        "target": target,
        "year": profile["year"],
        "official_template": True,
        "source_url": profile["url"],
        "author_guide": profile["guide"],
        "archive_sha256": archive_hash,
        "style_file": profile["style"],
        "copied_files": sorted(set(copied)),
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    (manuscript_root / "venue-template.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
