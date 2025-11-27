from __future__ import annotations

import logging
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
    MCPClientError,
    ProviderListPayload,
    ProviderRecord,
    ProviderScheduleResponse,
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
                    docs, audit = await self._handle_pricing(state, tool_audit)
                elif decision.intent in {"check_availability", "book_appointment"}:
                    docs, audit = await self._handle_schedule(state, tool_audit)
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

            tool_audit.extend(audit)
            span.set_attribute("workflow.context_docs", len(docs))

            time_context = qa_engine.build_time_context(question)
            qa_payload = qa_engine.answer_question(
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
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        clinic_id, provider_id, provider_entry = await self._resolve_entities(state)
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
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        clinic_id, provider_id, provider_entry = await self._resolve_entities(state)
        if not clinic_id:
            raise MCPWorkflowError(
                "محتاج أعرف اسم العيادة عشان أقدر أجيب المواعيد.",
                reason="missing_clinic",
            )

        schedule_response = await self._client.get_clinic_provider_schedule(
            clinic_id=clinic_id,
            provider_id=provider_id,
            day_id=None,
        )
        audit.append(
            ToolAuditEntry(
                name="get_clinic_provider_schedule",
                status="success",
                details={
                    "clinic_id": clinic_id,
                    "provider_id": provider_id,
                    "slots": len(schedule_response.slots),
                },
            )
        )

        if not schedule_response.slots:
            raise MCPWorkflowError(
                "مفيش مواعيد متاحة دلوقتي في النظام.",
                reason="empty_schedule_response",
            )

        context_text = _format_schedule(schedule_response, provider_entry)
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
    ) -> Tuple[Optional[int], Optional[int], Optional[ProviderRecord]]:
        """Resolve clinic and provider identifiers using MCP data with fuzzy matching."""

        doctor_name = state.entities.doctor
        clinic_name = state.entities.clinic

        provider_entry: Optional[ProviderRecord] = None
        
        if doctor_name:
            # Try exact match first (fast path)
            provider_entry = await self._client.lookup_provider_record(doctor_name, clinic_name=clinic_name)
            
            # If no exact match, try fuzzy matching
            if not provider_entry:
                provider_list = await self._client.get_clinic_provider_list()
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
            # Try exact match first
            clinic_entry = await self._client.lookup_clinic_record(clinic_name)
            
            # If no exact match, try fuzzy matching
            if not clinic_entry:
                provider_list = await self._client.get_clinic_provider_list()
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

