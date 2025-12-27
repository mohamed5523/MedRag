from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import trace
from pydantic import BaseModel, Field

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - fallback for older langchain versions
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        from langchain.docstore.document import Document  # type: ignore

from app.core.intent_router import RouteDecision, RouteMode
from app.core.qa_engine import QAEngine
from app.core.state_manager import ConversationState
from app.integrations.mcp_client import (
    DoctorMatchResult,
    HybridMatchResponse,
    HybridMatchStatus,
    MCPClient,
    ProviderListPayload,
    ProviderRecord,
    ProviderScheduleResponse,
    ScheduleSlot,
    ServicePriceResponse,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.clinic_workflow")


class MCPWorkflowError(Exception):
    """Raised when the clinic workflow cannot be completed."""

    def __init__(self, message: str, *, reason: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.data = data or {}


class ToolAuditEntry(BaseModel):
    name: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ClinicWorkflowResult(BaseModel):
    qa_response: Dict[str, Any]
    tool_audit: List[ToolAuditEntry] = Field(default_factory=list)


class ClinicWorkflowService:
    """Coordinates multi-step MCP tool usage and prepares LLM-friendly context."""

    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self._client = mcp_client or MCPClient()

    async def aclose(self):
        """Close the underlying MCP client."""
        await self._client.aclose()

    async def run(
        self,
        *,
        decision: RouteDecision,
        state: ConversationState,
        question: str,
        qa_engine: QAEngine,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ClinicWorkflowResult:
        if decision.mode != RouteMode.MCP:
            raise ValueError("ClinicWorkflowService can only execute MCP decisions.")

        with tracer.start_as_current_span("clinic_workflow.run") as span:
            span.set_attribute("workflow.intent", decision.intent)
            span.set_attribute("workflow.requires_provider_resolution", decision.requires_provider_resolution)
            span.set_attribute("workflow.tool_plan.count", len(decision.tool_sequence))

            tool_audit: List[ToolAuditEntry] = []

            try:
                if decision.intent == "ask_price":
                    docs, audit = await self._handle_pricing(state, tool_audit, question)
                elif decision.intent in {"check_availability", "book_appointment"}:
                    docs, audit = await self._handle_schedule(state, tool_audit, question)
                elif decision.intent == "list_doctors":
                    docs, audit = await self._handle_list_doctors(state, tool_audit)
                else:
                    raise MCPWorkflowError(
                        f"Intent {decision.intent} is not supported by the clinic workflow.",
                        reason="intent_not_supported",
                    )
            except MCPWorkflowError as exc:
                span.record_exception(exc)
                span.set_attribute("workflow.failure_reason", exc.reason)
                raise
            # Attach tool audit and MCP-derived context previews so they are visible in Phoenix
            tool_audit.extend(audit)
            span.set_attribute("workflow.context_docs", len(docs))
            try:
                if docs:
                    combined_context = "\n\n".join(doc.page_content for doc in docs)
                    span.set_attribute("workflow.mcp_context_preview", combined_context[:1000])
                    span.set_attribute(
                        "workflow.mcp_context_sources",
                        [doc.metadata.get("source", "Unknown") for doc in docs],
                    )
            except Exception:
                # Never break the workflow because of tracing/serialization issues
                logger.debug("Failed to attach MCP context preview to span", exc_info=True)

            time_context = qa_engine.build_time_context(question)
            qa_payload = await qa_engine.answer_question(
                question=question,
                contexts=docs,
                time_context=time_context,
                chat_history=chat_history,
            )

            return ClinicWorkflowResult(
                qa_response=qa_payload,
                tool_audit=tool_audit,
            )

    async def _handle_pricing(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        clinic_id, provider_id, provider_entry = await self._resolve_entities(state, audit, question)
        if not clinic_id:
            raise MCPWorkflowError(
                "لم أقدر أحدد العيادة المطلوبة عشان أجيب الأسعار.",
                reason="missing_clinic",
            )

        price_response = await self._client.get_service_price(clinic_id=clinic_id, provider_id=provider_id)
        audit.append(
            ToolAuditEntry(
                name="get_service_price",
                status="success",
                details={
                    "clinic_id": clinic_id,
                    "provider_id": provider_id,
                    "services_found": len(price_response.services),
                },
            )
        )

        if not price_response.services:
            raise MCPWorkflowError(
                "مفيش أسعار متاحة في السيستم للطلب ده حالياً.",
                reason="empty_price_response",
            )

        context_text = _format_service_prices(price_response, provider_entry)
        doc = Document(
            page_content=context_text,
            metadata={"source": "mcp.service_price"},
        )
        return [doc], audit

    async def _handle_schedule(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        clinic_id, provider_id, provider_entry = await self._resolve_entities(
            state, audit, question
        )
        if not clinic_id:
            raise MCPWorkflowError(
                "محتاج أعرف اسم العيادة عشان أقدر أجيب المواعيد.",
                reason="missing_clinic",
            )

        # For availability queries, call the schedule tool for each day of the week
        # and aggregate all returned slots into a single unified schedule view.
        all_slots: List[ScheduleSlot] = []

        # Create tasks for all days to run in parallel
        tasks = [
            self._client.get_clinic_provider_schedule(
                clinic_id=clinic_id,
                provider_id=provider_id,
                day_id=day_id,
            )
            for day_id in range(1, 8)  # 1=Saturday .. 7=Friday (see DAY_NAME_TO_ID)
        ]
        
        # Execute all requests concurrently
        day_responses = await asyncio.gather(*tasks)

        for day_id, day_response in enumerate(day_responses, start=1):
            audit.append(
                ToolAuditEntry(
                    name="get_clinic_provider_schedule",
                    status="success",
                    details={
                        "clinic_id": clinic_id,
                        "provider_id": provider_id,
                        "day_id": day_id,
                        "slots": len(day_response.slots),
                    },
                )
            )
            if day_response.slots:
                all_slots.extend(day_response.slots)

        if not all_slots:
            raise MCPWorkflowError(
                "مفيش مواعيد متاحة دلوقتي في النظام.",
                reason="empty_schedule_response",
            )

        aggregated_schedule = ProviderScheduleResponse(slots=all_slots)
        context_text = _format_schedule(aggregated_schedule, provider_entry)
        doc = Document(
            page_content=context_text,
            metadata={"source": "mcp.provider_schedule"},
        )
        return [doc], audit

    async def _handle_list_doctors(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        provider_list = await self._client.get_clinic_provider_list()
        audit.append(
            ToolAuditEntry(
                name="get_clinic_provider_list",
                status="success",
                details={"providers": len(provider_list.providers)},
            )
        )

        filtered = _filter_provider_list(
            provider_list,
            clinic_name=state.entities.clinic,
            specialty=state.entities.specialty,
        )
        if not filtered.providers:
            raise MCPWorkflowError(
                "ملقتش دكاترة مطابقين للعيادة أو التخصص اللي اتذكروا.",
                reason="provider_not_found",
            )

        context_text = _format_provider_list(filtered)
        doc = Document(page_content=context_text, metadata={"source": "mcp.provider_list"})
        return [doc], audit

    async def _resolve_entities(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        raw_question: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[int], Optional[ProviderRecord]]:
        """
        Resolve clinic and provider identifiers using MCP hybrid matching.
        
        Uses the new match_doctor_hybrid MCP tool for advanced name matching with:
        - Token-based matching with positional weights
        - Fuzzy matching for typos and variations
        - Arabic/English name support
        """
        doctor_name = state.entities.doctor
        clinic_name = state.entities.clinic
        question_text = state.last_user_question or raw_question or ""

        # Fallback heuristics if entity extraction missed them
        if not doctor_name:
            inferred_doctor = _infer_doctor_from_text(question_text)
            if inferred_doctor:
                logger.debug("Inferred doctor name '%s' from question text.", inferred_doctor)
                doctor_name = inferred_doctor

        if not clinic_name:
            inferred_clinic = _infer_clinic_from_context(state, question_text)
            if inferred_clinic:
                logger.debug("Inferred clinic name '%s' from state/question context.", inferred_clinic)
                clinic_name = inferred_clinic

        provider_entry: Optional[ProviderRecord] = None
        clinic_id: Optional[int] = None
        provider_id: Optional[int] = None
        
        if doctor_name:
            # Use the new hybrid matching MCP tool
            with tracer.start_as_current_span("mcp.match_doctor_hybrid") as span:
                span.set_attribute("query.doctor_name", doctor_name)
                if clinic_name:
                    span.set_attribute("query.clinic_name", clinic_name)
                
                match_response = await self._client.match_doctor_hybrid(
                    query=doctor_name,
                    clinic_name=clinic_name,
                    top_k=5,
                    min_score_multi=0.6,
                    min_score_single=0.55,
                )
                
                audit.append(
                    ToolAuditEntry(
                        name="match_doctor_hybrid",
                        status="success",
                        details={
                            "query": doctor_name,
                            "clinic_name": clinic_name,
                            "status": match_response.status.value,
                            "candidates_count": len(match_response.candidates),
                            "best_match_score": match_response.best_match.score if match_response.best_match else None,
                        },
                    )
                )
                
                span.set_attribute("match.status", match_response.status.value)
                span.set_attribute("match.candidates_count", len(match_response.candidates))
                
                if match_response.status == HybridMatchStatus.UNAMBIGUOUS_MATCH:
                    # Validate that best_match is present for UNAMBIGUOUS_MATCH
                    if not match_response.best_match:
                        logger.error(
                            "MCP returned UNAMBIGUOUS_MATCH without best_match object - invalid response"
                        )
                        raise MCPWorkflowError(
                            "حصل خطأ في نظام البحث. من فضلك حاول تاني.",
                            reason="invalid_match_response",
                        )
                    
                    # Clear match found - use it
                    best = match_response.best_match
                    logger.info(f"Hybrid match found: {best.name_ar or best.name_en} (score: {best.score:.2f})")
                    
                    # Safely convert string IDs to integers, handling non-numeric values
                    parsed_provider_id = _safe_parse_int(best.provider_id)
                    parsed_clinic_id = _safe_parse_int(best.clinic_id)
                    
                    # Convert DoctorMatchResult to ProviderRecord for compatibility
                    provider_entry = ProviderRecord(
                        provider_id=parsed_provider_id,
                        clinic_id=parsed_clinic_id,
                        provider_name_ar=best.name_ar,
                        provider_name_en=best.name_en,
                        clinic_name_ar=best.clinic_name,
                        clinic_name_en=best.clinic_name,
                    )
                    clinic_id = parsed_clinic_id
                    provider_id = parsed_provider_id
                    
                elif match_response.status == HybridMatchStatus.AMBIGUOUS_NEED_MORE_INFO:
                    # Multiple matches - ask user for clarification
                    alt_names = [
                        candidate.name_ar or candidate.name_en
                        for candidate in match_response.candidates[:3]
                    ]
                    if alt_names:
                        raise MCPWorkflowError(
                            f"يوجد أكثر من دكتور بنفس الاسم. هل تقصد: {' أو '.join(alt_names)}؟",
                            reason="provider_ambiguous",
                            data={
                                "candidates": [
                                    {
                                        "provider_id": _safe_parse_int(c.provider_id),
                                        "clinic_id": _safe_parse_int(c.clinic_id),
                                        "clinic_name": c.clinic_name,
                                        "name_ar": c.name_ar,
                                        "name_en": c.name_en,
                                        "score": c.score,
                                    }
                                    for c in match_response.candidates
                                ]
                            },
                        )
                    else:
                        raise MCPWorkflowError(
                            match_response.message,
                            reason="provider_ambiguous",
                            data={"candidates": []},
                        )
                        
                elif match_response.status == HybridMatchStatus.NO_MATCH:
                    # No match found
                    raise MCPWorkflowError(
                        match_response.message or "ملقتش دكتور بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                        reason="provider_not_found",
                    )
                    
                elif match_response.status == HybridMatchStatus.LOW_CONFIDENCE:
                    # Low confidence match - treat similar to ambiguous, ask for clarification
                    alt_names = [
                        candidate.name_ar or candidate.name_en
                        for candidate in match_response.candidates[:3]
                    ]
                    if alt_names:
                        raise MCPWorkflowError(
                            f"مش متأكد من الاسم. هل تقصد: {' أو '.join(alt_names)}؟",
                            reason="provider_low_confidence",
                            data={
                                "candidates": [
                                    {
                                        "provider_id": _safe_parse_int(c.provider_id),
                                        "clinic_id": _safe_parse_int(c.clinic_id),
                                        "clinic_name": c.clinic_name,
                                        "name_ar": c.name_ar,
                                        "name_en": c.name_en,
                                        "score": c.score,
                                    }
                                    for c in match_response.candidates
                                ]
                            },
                        )
                    else:
                        raise MCPWorkflowError(
                            match_response.message or "ملقتش دكتور بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                            reason="provider_low_confidence",
                            data={"candidates": []},
                        )
                        
                else:
                    # Unknown/unexpected status - log and raise error
                    logger.error(
                        f"MCP returned unexpected match status: {match_response.status}"
                    )
                    raise MCPWorkflowError(
                        "حصل خطأ في نظام البحث. من فضلك حاول تاني.",
                        reason="unexpected_match_status",
                    )

        # If we have a clinic name but no doctor (or doctor resolution gave us clinic info)
        if not clinic_id and clinic_name:
            # Fetch provider list to find clinic
            with tracer.start_as_current_span("mcp.get_clinic_provider_list") as span:
                provider_list = await self._client.get_clinic_provider_list()
                span.set_attribute("providers.count", len(provider_list.providers))
                
                audit.append(
                    ToolAuditEntry(
                        name="get_clinic_provider_list",
                        status="success",
                        details={"providers_count": len(provider_list.providers)},
                    )
                )
            
            # Try exact match for clinic
            clinic_entry = provider_list.find_clinic(clinic_name)
            if clinic_entry:
                clinic_id = clinic_entry.clinic_id
                logger.info(f"Found clinic by exact match: {clinic_entry.clinic_name_ar}")
            else:
                # No exact clinic match - inform user
                raise MCPWorkflowError(
                    "ملقتش عيادة بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                    reason="clinic_not_found",
                )

        return clinic_id, provider_id, provider_entry


def _safe_parse_int(value: Any) -> Optional[int]:
    """
    Safely parse a value to int, returning None for invalid/non-numeric values.
    
    Handles cases where the MCP server returns:
    - None or empty string
    - String "None" or "null"
    - Non-numeric strings
    - Valid numeric strings or integers
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Handle string representations of null/None
        if value.strip().lower() in ("", "none", "null"):
            return None
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Failed to parse '{value}' as integer, returning None")
            return None
    # For any other type, try conversion
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse '{value}' (type: {type(value).__name__}) as integer, returning None")
        return None


def _infer_doctor_from_text(text: str) -> Optional[str]:
    """
    Extract doctor name heuristically from user text, e.g. "دكتور ابانوب".
    Supports capturing up to 3 tokens after the keyword.
    """
    if not text:
        return None

    normalized = text.replace("ـ", "").strip()
    # Capture sequences after "دكتور" / "د." / "د "
    doctor_patterns = [
        r"(?:دكتور|د\.?|د\s+)\s+([\u0621-\u064A\w]+(?:\s+[\u0621-\u064A\w]+){0,2})",
    ]
    for pattern in doctor_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            candidate = _trim_punctuation(candidate)
            candidate = _strip_leading_doctor_words(candidate)
            candidate = _strip_trailing_clinic_words(candidate)
            if candidate:
                return candidate
    return None


def _infer_clinic_from_context(state: ConversationState, question_text: str) -> Optional[str]:
    """
    Infer clinic name using specialty or raw question text when entity extraction misses it.
    """
    specialty = state.entities.specialty if state and state.entities else None
    if specialty:
        normalized = specialty.strip()
        if normalized:
            if normalized.startswith("عيادة"):
                return normalized
            return f"عيادة {normalized}".strip()

    inferred = _infer_phrase_after_keywords(
        question_text,
        keywords=["عيادة", "عياده", "clinic"],
        max_words=4,
    )
    return inferred


def _infer_phrase_after_keywords(
    text: str,
    keywords: List[str],
    max_words: int = 3,
) -> Optional[str]:
    """
    Given text, extract the phrase immediately following any of the provided keywords.
    Useful for capturing "عيادة الأسنان" or "clinic oncology".
    """
    if not text:
        return None

    sanitized = text.replace("؟", "?").replace("،", ",")
    for keyword in keywords:
        pattern = rf"{keyword}\s+([\u0621-\u064A\w\s]+)"
        match = re.search(pattern, sanitized, flags=re.IGNORECASE)
        if not match:
            continue

        candidate = match.group(1)
        candidate = re.split(r"[?.,!،]", candidate)[0]
        words = candidate.split()
        candidate = " ".join(words[:max_words]).strip()
        if not candidate:
            continue

        # Ensure Arabic keywords keep the "عيادة" prefix for better matching
        keyword_lower = keyword.lower()
        if keyword_lower.startswith("عي"):
            if not candidate.startswith("عيادة"):
                candidate = f"عيادة {candidate}".strip()
            else:
                candidate = candidate
        return candidate

    return None


def _trim_punctuation(value: str) -> str:
    """Remove trailing punctuation and extra whitespace from a captured phrase."""
    return value.strip(" \t\n\r?.,!،") if value else value


def _strip_leading_doctor_words(value: str) -> str:
    """Remove leading tokens such as 'دكتور' or 'Dr' from a captured phrase."""
    if not value:
        return value
    tokens = value.split()
    while tokens:
        head = tokens[0].replace("ـ", "")
        head_lower = head.casefold().strip(".")
        if head_lower in {"دكتور", "دكتوره", "د", "dr", "doctor"}:
            tokens.pop(0)
            continue
        break
    return " ".join(tokens)


def _strip_trailing_clinic_words(value: str) -> str:
    """Remove trailing words such as 'لعيادة' that leak into doctor captures."""
    if not value:
        return value
    tokens = value.split()
    while tokens:
        tail = tokens[-1].replace("ـ", "")
        tail_lower = tail.casefold()
        if any(
            keyword in tail_lower
            for keyword in ("عياد", "clinic")
        ):
            tokens.pop()
            continue
        break
    return " ".join(tokens)


def _format_service_prices(
    response: ServicePriceResponse,
    provider_entry: Optional[ProviderRecord],
) -> str:
    doctor_name = (
        (provider_entry.provider_name_ar or provider_entry.provider_name_en)
        if provider_entry
        else None
    )
    clinic_name = (
        (provider_entry.clinic_name_ar or provider_entry.clinic_name_en)
        if provider_entry
        else None
    )

    header_parts = ["بيانات الأسعار الرسمية من النظام."]
    if clinic_name:
        header_parts.append(f"العيادة: {clinic_name}.")
    if doctor_name:
        header_parts.append(f"الدكتور: {doctor_name}.")

    lines = header_parts + ["\nالخدمات:"]
    for service in response.services:
        name = service.service_name_ar or service.service_name_en or "خدمة بدون اسم"
        price = f"{service.price:.2f}" if service.price is not None else "غير متاح"
        currency = service.currency or "EGP"
        lines.append(f"- {name}: {price} {currency}")

    return "\n".join(lines)


def _format_schedule(
    response: ProviderScheduleResponse,
    provider_entry: Optional[ProviderRecord],
) -> str:
    doctor_name = (
        (provider_entry.provider_name_ar or provider_entry.provider_name_en)
        if provider_entry
        else None
    )
    clinic_name = (
        (provider_entry.clinic_name_ar or provider_entry.clinic_name_en)
        if provider_entry
        else None
    )

    header = ["جبتلك المواعيد الرسمية من النظام."]
    if clinic_name:
        header.append(f"العيادة: {clinic_name}.")
    if doctor_name:
        header.append(f"الدكتور: {doctor_name}.")

    grouped: Dict[str, List[str]] = defaultdict(list)
    for slot in response.slots:
        day = slot.day_name or f"اليوم #{slot.day_id}"
        start = slot.shift_start or "غير محدد"
        end = slot.shift_end or "غير محدد"
        grouped[day].append(f"{start} → {end}")

    for day, entries in grouped.items():
        header.append(f"\n{day}:")
        for entry in entries:
            header.append(f"- {entry}")

    return "\n".join(header)


def _format_provider_list(provider_list: ProviderListPayload) -> str:
    lines = ["قائمة الدكاترة المتاحة حسب بيانات السيستم:"]
    for record in provider_list.providers:
        doctor = record.provider_name_ar or record.provider_name_en or "دكتور"
        clinic = record.clinic_name_ar or record.clinic_name_en or "عيادة غير معروفة"
        specialty = record.specialty or "التخصص غير محدد"
        lines.append(f"- {doctor} في {clinic} — تخصص: {specialty}")
    return "\n".join(lines)


def _filter_provider_list(
    provider_list: ProviderListPayload,
    *,
    clinic_name: Optional[str],
    specialty: Optional[str],
) -> ProviderListPayload:
    filtered: List[ProviderRecord] = []
    for record in provider_list.providers:
        if clinic_name and not record.matches_clinic(clinic_name):
            continue
        if specialty and record.specialty and specialty.casefold() not in record.specialty.casefold():
            continue
        filtered.append(record)
    return ProviderListPayload(providers=filtered or provider_list.providers[:10])

