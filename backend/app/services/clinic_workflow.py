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

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


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
        Resolve clinic and provider identifiers using MCP data with fuzzy matching.
        
        Optimized to fetch provider list ONCE and reuse for all lookups (exact and fuzzy).
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

        # ✅ Fetch provider list ONCE - all subsequent operations use this in-memory data
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

        provider_entry: Optional[ProviderRecord] = None
        
        if doctor_name:
            # Try exact match first (in-memory, no API call)
            provider_entry = provider_list.find_provider(doctor_name, clinic_name=clinic_name)
            
            # If no exact match, try fuzzy matching (in-memory, no API call)
            if not provider_entry:
                fuzzy_matches = provider_list.find_provider_fuzzy(
                    doctor_name, 
                    clinic_name=clinic_name,
                    top_k=3
                )
                
                if fuzzy_matches:
                    best_match, confidence = fuzzy_matches[0]
                    
                    if confidence >= 0.85:
                        # Very high confidence - use it with a note
                        logger.info(f"Auto-selected provider {best_match.provider_name_ar} with confidence {confidence:.2f}")
                        provider_entry = best_match
                        
                    elif confidence >= 0.65:
                        # Good confidence - use it but inform user
                        logger.info(f"Using provider {best_match.provider_name_ar} with confidence {confidence:.2f}")
                        provider_entry = best_match
                        
                    else:
                        # Low confidence - show alternatives
                        alt_names = [
                            f"{match.provider_name_ar or match.provider_name_en}"
                            for match, score in fuzzy_matches[:3]
                            if score > 0.4
                        ]
                        
                        if alt_names:
                            raise MCPWorkflowError(
                                f"مفيش دكتور بالاسم ده بالضبط. هل تقصد: {' أو '.join(alt_names)}؟",
                                reason="provider_ambiguous",
                            )
                        else:
                            raise MCPWorkflowError(
                                "ملقتش دكتور بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                                reason="provider_not_found",
                            )

        clinic_entry: Optional[ProviderRecord] = None
        if provider_entry:
            clinic_entry = provider_entry
        elif clinic_name:
            # Try exact match first (in-memory, no API call)
            clinic_entry = provider_list.find_clinic(clinic_name)
            
            # If no exact match, try fuzzy matching (in-memory, no API call)
            if not clinic_entry:
                fuzzy_matches = provider_list.find_clinic_fuzzy(clinic_name, top_k=3)
                
                if fuzzy_matches:
                    best_match, confidence = fuzzy_matches[0]
                    
                    if confidence >= 0.75:
                        logger.info(f"Using clinic {best_match.clinic_name_ar} with confidence {confidence:.2f}")
                        clinic_entry = best_match
                    else:
                        alt_names = [
                            match.clinic_name_ar or match.clinic_name_en
                            for match, score in fuzzy_matches[:3]
                            if score > 0.4
                        ]
                        if alt_names:
                            raise MCPWorkflowError(
                                f"مفيش عيادة بالاسم ده بالضبط. هل تقصد: {' أو '.join(alt_names)}؟",
                                reason="clinic_ambiguous",
                            )
                        else:
                            # All matches below display threshold
                            raise MCPWorkflowError(
                                "ملقتش عيادة بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                                reason="clinic_not_found",
                            )

        clinic_id = clinic_entry.clinic_id if clinic_entry else None
        provider_id = provider_entry.provider_id if provider_entry else None

        return clinic_id, provider_id, provider_entry


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

