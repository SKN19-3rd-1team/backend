"""
전공 데이터 로딩 모듈

major_detail.json 파일에서 전공 정보를 읽어서 MajorRecord 객체로 변환합니다.
전공 추천 시스템에서 사용됩니다.
"""

from __future__ import annotations

# backend/rag/loader.py
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from backend.config import get_settings, resolve_path


# ==================== Major Detail Loading ====================

# JSON 전공 데이터를 구조화한 레코드 모델
@dataclass
class MajorRecord:
    """
    backend/data/major_detail.json에서 파싱한 전공 원본 레코드를 표현한다.
    """

    major_id: str
    major_name: str
    cluster: Optional[str]
    summary: str
    interest: str
    property: str
    relate_subject: Any
    job: str
    enter_field: Any
    salary: Optional[float]
    employment: Optional[str] = None
    gender: Any = None
    satisfaction: Any = None
    employment_rate: Any = None
    acceptance_rate: Optional[float] = None
    department_aliases: list[str] = field(default_factory=list)
    career_act: Any = None
    qualifications: Any = None
    main_subject: Any = None
    university: Any = None
    chart_data: Any = None
    raw: dict = field(default_factory=dict, repr=False)


# Pinecone에 업로드할 전공 문서 구조
@dataclass
class MajorDoc:
    """
    Pinecone 인덱싱을 위해 전처리된 구조화 문서 표현체.
    """

    doc_id: str
    major_id: str
    major_name: str
    doc_type: str
    text: str
    cluster: Optional[str] = None
    salary: Optional[float] = None
    relate_subject_tags: list[str] = field(default_factory=list)
    job_tags: list[str] = field(default_factory=list)
    raw_subjects: Optional[str] = None
    raw_jobs: Optional[str] = None


def _slugify(value: str) -> str:
    # 전공명을 Pinecone 문서 ID로 활용하기 위해 안전한 슬러그 형태로 변환
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug


def _normalize_whitespace(text: str) -> str:
    # 공백이나 줄바꿈을 하나의 공백으로 단일화
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(value: str) -> str:
    # 간단한 HTML 태그를 제거하여 텍스트만 남김
    return re.sub(r"<[^>]+>", " ", value or "")


def _parse_salary(raw_salary: Any) -> Optional[float]:
    # 급여 필드가 문자열/숫자 등 다양한 형태로 올 수 있어 float로 정규화
    if raw_salary is None:
        return None
    if isinstance(raw_salary, (int, float)):
        return float(raw_salary)
    if isinstance(raw_salary, str):
        digits = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw_salary.replace(",", ""))
        if digits:
            return float(digits[0])
    return None


def _calculate_acceptance_rate(chart_data: Any) -> Optional[float]:
    """
    chartData의 applicant 정보를 기반으로 합격률(입학자/지원자 * 100)을 계산한다.
    구조: [{"item": "지원자", "data": "21525"}, {"item": "입학자", "data": "3448"}]
    """
    if not chart_data or not isinstance(chart_data, list) or not chart_data:
        return None
    
    # chartData는 리스트이고 첫 번째 요소에 모든 데이터가 있음
    stats = chart_data[0]
    if not isinstance(stats, dict):
        return None
        
    applicants_list = stats.get("applicant")
    if not applicants_list or not isinstance(applicants_list, list):
        return None

    applicant_count = 0.0
    entrant_count = 0.0

    for item in applicants_list:
        if not isinstance(item, dict):
            continue
        label = item.get("item")
        data_str = item.get("data", "0")
        
        try:
            val = float(data_str)
            if label == "지원자":
                applicant_count = val
            elif label == "입학자":
                entrant_count = val
        except (ValueError, TypeError):
            continue

    if applicant_count > 0:
        rate = (entrant_count / applicant_count) * 100
        return round(rate, 1)
        
    return None


def _split_multi_value(value: str) -> list[str]:
    # 콤마/슬래시 등으로 구분된 학과명 문자열을 리스트로 변환
    if not value:
        return []
    parts = re.split(r"[,/;]", value)
    cleaned = []
    for part in parts:
        token = part.strip()
        if token:
            cleaned.append(token)
    return cleaned


def load_major_detail(path: str | Path | None = None) -> list[MajorRecord]:
    """
    major_detail.json 파일을 읽어 MajorRecord 객체 리스트로 변환한다.

    Args:
        path: 기본 경로를 덮어쓰고 싶을 때 사용하는 커스텀 JSON 경로

    Returns:
        MajorRecord 리스트 (전공별 원본 데이터 포함)
    """
    # 기본 경로는 설정값(MAJOR_DETAIL_PATH)을 사용
    settings = get_settings()
    json_path = Path(resolve_path(path if path else settings.major_detail_path))
    data = json.loads(json_path.read_text(encoding="utf-8"))

    records: list[MajorRecord] = []
    seen_ids: dict[str, int] = {}

    for block_index, block in enumerate(data):
        contents: Sequence[dict[str, Any]] = block.get("dataSearch", {}).get("content", []) or []
        for content_index, payload in enumerate(contents):
            major_name = (payload.get("major") or "").strip()
            base_slug = _slugify(major_name) or f"major-{block_index}-{content_index}"
            dedup_idx = seen_ids.get(base_slug, 0)
            seen_ids[base_slug] = dedup_idx + 1
            major_id = base_slug if dedup_idx == 0 else f"{base_slug}-{dedup_idx}"

            salary = _parse_salary(payload.get("salary"))
            department_aliases = _split_multi_value(payload.get("department", ""))
            
            # chartData 처리
            chart_data = payload.get("chartData")
            acceptance_rate = _calculate_acceptance_rate(chart_data)
            
            # 통계 데이터 추출 (chartData[0] 내부에 존재)
            gender_stats = None
            satisfaction_stats = None
            employment_rate_stats = None
            
            if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
                stats_block = chart_data[0]
                if isinstance(stats_block, dict):
                    gender_stats = stats_block.get("gender")
                    satisfaction_stats = stats_block.get("satisfaction")
                    employment_rate_stats = stats_block.get("employment_rate")

            record = MajorRecord(
                major_id=major_id,
                major_name=major_name or f"미확인 전공 {len(records) + 1}",
                cluster=payload.get("cluster"),
                summary=(payload.get("summary") or "").strip(),
                interest=(payload.get("interest") or "").strip(),
                property=(payload.get("property") or "").strip(),
                relate_subject=payload.get("relate_subject"),
                job=(payload.get("job") or "").strip(),
                enter_field=payload.get("enter_field"),
                salary=salary,
                employment=payload.get("employment"),
                gender=gender_stats,
                satisfaction=satisfaction_stats,
                employment_rate=employment_rate_stats,
                acceptance_rate=acceptance_rate,
                department_aliases=department_aliases,
                career_act=payload.get("career_act"),
                qualifications=payload.get("qualifications"),
                main_subject=payload.get("main_subject"),
                university=payload.get("university"),
                chart_data=chart_data,
                raw=payload,
            )
            records.append(record)

    return records


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    # Pinecone 메타데이터에 사용할 태그를 순서 유지한 채 중복 제거
    seen = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _extract_subject_tags(relate_subject: Any) -> list[str]:
    tags: list[str] = []
    if not isinstance(relate_subject, list):
        return tags

    for item in relate_subject:
        description = _strip_html(item.get("subject_description") or "")
        description = re.sub(r":", " ", description)
        parts = re.split(r"[,/·ㆍ\n]", description)
        for part in parts:
            candidate = part.strip()
            if len(candidate) < 2:
                continue
            tags.append(candidate)

    return _unique_preserve_order(tags)


def _format_subject_text(relate_subject: Any) -> tuple[Optional[str], list[str], Optional[str]]:
    # 과목 리스트를 사람이 읽을 수 있는 텍스트와 태그 목록으로 변환
    if not isinstance(relate_subject, list):
        return None, [], None

    lines: list[str] = []
    for subject in relate_subject:
        name = (subject.get("subject_name") or "").strip()
        description = _strip_html(subject.get("subject_description") or "").strip()
        if not name and not description:
            continue
        if description:
            lines.append(f"{name}: {description}" if name else description)
        else:
            lines.append(name)

    if not lines:
        return None, [], None

    text = (
        "관련 과목 안내:\n" + "\n".join(f"- {line}" for line in lines)
    )
    raw_block = "\n".join(lines)
    tags = _extract_subject_tags(relate_subject)
    return text, tags, raw_block


def _extract_job_tags(job_text: str) -> list[str]:
    # 진출 직업 문자열에서 직업명을 태그로 추출
    tags = []
    parts = re.split(r"[,\n/]", job_text or "")
    for part in parts:
        candidate = part.strip()
        if len(candidate) < 2:
            continue
        tags.append(candidate)
    return _unique_preserve_order(tags)


def _format_job_text(job_text: str, enter_field: Any) -> tuple[Optional[str], list[str], Optional[str]]:
    # 직업/진출분야 정보를 하나의 설명 텍스트로 합치고 태그를 뽑아냄
    lines: list[str] = []
    if job_text:
        lines.append(f"주요 진출 직업: {job_text}")

    if isinstance(enter_field, list):
        for item in enter_field:
            category = (item.get("gradeuate") or item.get("graduate") or "").strip()
            description = _strip_html(item.get("description") or "").strip()
            if category and description:
                lines.append(f"{category}: {description}")
            elif description:
                lines.append(description)

    if not lines:
        return None, [], None

    combined_text = "\n".join(lines)
    tags = _extract_job_tags(job_text)
    return combined_text, tags, combined_text


def build_major_docs(record: MajorRecord) -> list[MajorDoc]:
    """
    단일 전공 레코드를 다양한 doc_type(summary, interest 등)으로 분해해 MajorDoc 리스트를 생성한다.
    """
    docs: list[MajorDoc] = []

    def make_doc(doc_type: str, text: str, **extra) -> MajorDoc:
        # doc_type별 공통 필드를 세팅하고 MajorDoc 인스턴스를 생성
        payload = MajorDoc(
            doc_id=f"{record.major_id}:{doc_type}",
            major_id=record.major_id,
            major_name=record.major_name,
            doc_type=doc_type,
            text=text.strip(),
            cluster=record.cluster,
            salary=record.salary,
            relate_subject_tags=extra.get("relate_subject_tags", []),
            job_tags=extra.get("job_tags", []),
            raw_subjects=extra.get("raw_subjects"),
            raw_jobs=extra.get("raw_jobs"),
        )
        return payload

    if record.summary:
        docs.append(make_doc("summary", record.summary))

    if record.interest:
        interest_text = record.interest
        # career activities can help describe interests further
        if isinstance(record.career_act, list):
            activities = []
            for act in record.career_act:
                name = (act.get("act_name") or "").strip()
                description = _strip_html(act.get("act_description") or "").strip()
                if name and description:
                    activities.append(f"{name}: {description}")
                elif description:
                    activities.append(description)
            if activities:
                interest_text = f"{interest_text}\n\n추천 활동:\n" + "\n".join(f"- {item}" for item in activities)
        docs.append(make_doc("interest", interest_text))

    if record.property:
        docs.append(make_doc("property", record.property))

    subjects_text, subject_tags, raw_subjects = _format_subject_text(record.relate_subject)
    if subjects_text:
        docs.append(
            make_doc(
                "subjects",
                subjects_text,
                relate_subject_tags=subject_tags,
                raw_subjects=raw_subjects,
            )
        )

    jobs_text, job_tags, raw_jobs = _format_job_text(record.job, record.enter_field)
    if jobs_text:
        docs.append(
            make_doc(
                "jobs",
                jobs_text,
                job_tags=job_tags,
                raw_jobs=raw_jobs,
            )
        )

    return docs


def build_all_major_docs(records: list[MajorRecord]) -> list[MajorDoc]:
    """
    전체 전공 레코드를 순회하며 build_major_docs 결과를 하나의 리스트로 합친다.
    """
    docs: list[MajorDoc] = []
    for record in records:
        docs.extend(build_major_docs(record))
    return docs
