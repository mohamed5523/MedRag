from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
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
    ClinicMatchResponse,
    DoctorMatchResult,
    DAY_NAME_TO_ID,
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
                time_context = qa_engine.build_time_context(question)
                now_dt = time_context.get("now_dt")
                effective_primary_intent = (
                    "who_is_present" if _is_who_is_present_query(question) else decision.intent
                )
                requested_intents = _detect_requested_intents(
                    question,
                    primary_intent=effective_primary_intent,
                    max_intents=3,
                )
                span.set_attribute("workflow.requested_intents", ",".join(requested_intents))
                span.set_attribute("workflow.effective_primary_intent", effective_primary_intent)

                docs: List[Document] = []

                # Preserve existing behavior for single-intent queries (errors should raise).
                if len(requested_intents) == 1:
                    intent = requested_intents[0]
                    if intent == "who_is_present":
                        docs, _ = await self._handle_who_is_present(
                            state, tool_audit, question, now_dt=now_dt
                        )
                    elif intent == "ask_price":
                        docs, _ = await self._handle_pricing(state, tool_audit, question)
                    elif intent in {"check_availability", "book_appointment"}:
                        docs, _ = await self._handle_schedule(state, tool_audit, question)
                    elif intent == "list_doctors":
                        docs, _ = await self._handle_list_doctors(state, tool_audit)
                    else:
                        raise MCPWorkflowError(
                            f"Intent {intent} is not supported by the clinic workflow.",
                            reason="intent_not_supported",
                        )
                else:
                    # Multi-intent execution: run up to 3 intent handlers and merge contexts into one QA call.
                    # If an intent requires disambiguation, bubble up the MCPWorkflowError so chat.py can store pending actions.
                    disambiguation_reasons = {
                        "provider_ambiguous",
                        "provider_low_confidence",
                        "clinic_ambiguous",
                        "provider_clinic_mismatch",
                    }

                    for intent in requested_intents:
                        try:
                            if intent == "who_is_present":
                                part_docs, _ = await self._handle_who_is_present(
                                    state, tool_audit, question, now_dt=now_dt
                                )
                            elif intent == "ask_price":
                                part_docs, _ = await self._handle_pricing(state, tool_audit, question)
                            elif intent in {"check_availability", "book_appointment"}:
                                part_docs, _ = await self._handle_schedule(state, tool_audit, question)
                            elif intent == "list_doctors":
                                part_docs, _ = await self._handle_list_doctors(state, tool_audit)
                            else:
                                raise MCPWorkflowError(
                                    f"Intent {intent} is not supported by the clinic workflow.",
                                    reason="intent_not_supported",
                                )
                            docs.extend(part_docs)
                        except MCPWorkflowError as exc:
                            if exc.reason in disambiguation_reasons:
                                raise
                            # Keep the workflow responsive: partial answer for other intents is better than failing the whole turn.
                            docs.append(
                                Document(
                                    page_content=f"[{intent}] {str(exc)}",
                                    metadata={"source": f"mcp.error.{intent}", "reason": exc.reason},
                                )
                            )
            except MCPWorkflowError as exc:
                span.record_exception(exc)
                span.set_attribute("workflow.failure_reason", exc.reason)
                raise
            # Attach tool audit and MCP-derived context previews so they are visible in Phoenix
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

        clinic_id: Optional[int] = getattr(state.entities, "clinic_id", None)
        if clinic_id is not None:
            filtered = _filter_provider_list_by_clinic_id(provider_list, clinic_id)
        else:
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

    async def _handle_who_is_present(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        now_dt: datetime | None,
        max_concurrency: int = 10,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        """
        Answer queries like: "مين موجود النهارده/بكرة في عيادة <X>؟"
        by joining provider list + provider schedules for the requested day only.
        """
        clinic_id, _, _ = await self._resolve_entities(state, audit, question)
        if not clinic_id:
            raise MCPWorkflowError(
                "محتاج أعرف اسم العيادة عشان أقدر أقول مين موجود.",
                reason="missing_clinic",
            )

        day_id = _infer_target_day_id(question, now_dt=now_dt)
        provider_list = await self._client.get_clinic_provider_list()
        audit.append(
            ToolAuditEntry(
                name="get_clinic_provider_list",
                status="success",
                details={"providers": len(provider_list.providers)},
            )
        )

        providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
        if not providers:
            # Fallback to name-based filtering if clinic_id not present in MCP payload for some reason.
            providers = _filter_provider_list(
                provider_list,
                clinic_name=state.entities.clinic,
                specialty=None,
            ).providers

        if not providers:
            raise MCPWorkflowError(
                "ملقتش دكاترة للعيادة دي في بيانات السيستم.",
                reason="provider_not_found",
            )

        sem = asyncio.Semaphore(max_concurrency)

        async def fetch_for_provider(p: ProviderRecord) -> tuple[ProviderRecord, ProviderScheduleResponse]:
            async with sem:
                resp = await self._client.get_clinic_provider_schedule(
                    clinic_id=clinic_id,
                    provider_id=p.provider_id,
                    day_id=day_id,
                )
                audit.append(
                    ToolAuditEntry(
                        name="get_clinic_provider_schedule",
                        status="success",
                        details={
                            "clinic_id": clinic_id,
                            "provider_id": p.provider_id,
                            "day_id": day_id,
                            "slots": len(resp.slots),
                        },
                    )
                )
                return p, resp

        results = await asyncio.gather(*(fetch_for_provider(p) for p in providers))

        present: list[tuple[str, list[str]]] = []
        for p, resp in results:
            if not resp.slots:
                continue
            name = p.provider_name_ar or p.provider_name_en or "دكتور"
            times = []
            for slot in resp.slots:
                start = slot.shift_start or "غير محدد"
                end = slot.shift_end or "غير محدد"
                times.append(f"{start} → {end}")
            present.append((name, times))

        if not present:
            raise MCPWorkflowError(
                "مفيش دكاترة عليهم مواعيد في اليوم ده حسب السيستم.",
                reason="empty_schedule_response",
            )

        clinic_label = state.entities.clinic or "العيادة"
        context_text = _format_who_is_present(
            clinic_label=clinic_label,
            day_id=day_id,
            present=present,
        )
        doc = Document(page_content=context_text, metadata={"source": "mcp.who_is_present"})
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
        clinic_id: Optional[int] = getattr(state.entities, "clinic_id", None)
        provider_id: Optional[int] = getattr(state.entities, "provider_id", None)

        # Fallback heuristics if entity extraction missed them
        if not doctor_name:
            inferred_doctor = _infer_doctor_from_text(question_text)
            if inferred_doctor:
                logger.debug("Inferred doctor name '%s' from question text.", inferred_doctor)
                doctor_name = inferred_doctor

        # Only infer a clinic name when the user did NOT provide a doctor name.
        # If a doctor is present, inferring clinic from specialty can incorrectly constrain
        # the match and cause repeated clinic disambiguation loops.
        if not clinic_name and not doctor_name and clinic_id is None:
            inferred_clinic = _infer_clinic_from_context(state, question_text)
            if inferred_clinic:
                logger.debug("Inferred clinic name '%s' from state/question context.", inferred_clinic)
                clinic_name = inferred_clinic

        provider_entry: Optional[ProviderRecord] = None

        # Resolve clinic_id early via hybrid clinic matcher when user mentions a clinic.
        # This makes the workflow robust against typos/variants in clinic names.
        if clinic_id is None and clinic_name:
            with tracer.start_as_current_span("mcp.match_clinic_hybrid") as span:
                span.set_attribute("query.clinic_name", clinic_name)
                clinic_match: ClinicMatchResponse = await self._client.match_clinic_hybrid(
                    query=clinic_name,
                    top_k=5,
                    min_score=0.65,
                )

                audit.append(
                    ToolAuditEntry(
                        name="match_clinic_hybrid",
                        status="success",
                        details={
                            "query": clinic_name,
                            "status": clinic_match.status.value,
                            "candidates_count": len(clinic_match.candidates),
                            "best_match_score": clinic_match.best_match.score if clinic_match.best_match else None,
                        },
                    )
                )

                span.set_attribute("match.status", clinic_match.status.value)
                span.set_attribute("match.candidates_count", len(clinic_match.candidates))

                if clinic_match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH:
                    if not clinic_match.best_match:
                        raise MCPWorkflowError(
                            "حصل خطأ في نظام البحث. من فضلك حاول تاني.",
                            reason="invalid_match_response",
                        )
                    parsed_clinic_id = _safe_parse_int(clinic_match.best_match.clinic_id)
                    if parsed_clinic_id is not None:
                        clinic_id = parsed_clinic_id
                    # Keep canonical clinic name for downstream prompts/logging
                    clinic_name = clinic_match.best_match.clinic_name or clinic_name
                    # Propagate canonical clinic into state so later turns + formatting use DB name.
                    try:
                        if hasattr(state, "entities") and hasattr(state.entities, "clinic"):
                            state.entities.clinic = clinic_name
                        if hasattr(state, "entities") and hasattr(state.entities, "clinic_id"):
                            state.entities.clinic_id = clinic_id
                    except Exception:
                        # Don't break workflow if state object is immutable in some contexts.
                        pass
                    # If this is a clinic-only query (no provider yet), create a lightweight
                    # ProviderRecord with the canonical clinic name so schedule/pricing formatting
                    # uses the DB name, not the user's typo.
                    if provider_entry is None:
                        provider_entry = ProviderRecord(
                            clinic_id=clinic_id,
                            clinic_name_ar=clinic_name,
                            clinic_name_en=clinic_name,
                        )

                elif clinic_match.status == HybridMatchStatus.AMBIGUOUS_NEED_MORE_INFO:
                    candidates = [
                        {
                            "clinic_id": _safe_parse_int(c.clinic_id),
                            "clinic_name": c.clinic_name,
                            "score": c.score,
                        }
                        for c in clinic_match.candidates[:5]
                    ]
                    raise MCPWorkflowError(
                        clinic_match.message or "يوجد أكثر من عيادة بنفس الاسم. من فضلك اختار العيادة.",
                        reason="clinic_ambiguous",
                        data={"candidates": candidates},
                    )

                elif clinic_match.status == HybridMatchStatus.NO_MATCH:
                    raise MCPWorkflowError(
                        clinic_match.message or "ملقتش عيادة بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
                        reason="clinic_not_found",
                    )
                # LOW_CONFIDENCE: treat like ambiguous
                elif clinic_match.status == HybridMatchStatus.LOW_CONFIDENCE:
                    candidates = [
                        {
                            "clinic_id": _safe_parse_int(c.clinic_id),
                            "clinic_name": c.clinic_name,
                            "score": c.score,
                        }
                        for c in clinic_match.candidates[:5]
                    ]
                    raise MCPWorkflowError(
                        clinic_match.message or "مش متأكد من اسم العيادة. ممكن تختار من اللي ظهروا؟",
                        reason="clinic_ambiguous",
                        data={"candidates": candidates},
                    )
        
        if doctor_name:
            # Use the new hybrid matching MCP tool
            with tracer.start_as_current_span("mcp.match_doctor_hybrid") as span:
                span.set_attribute("query.doctor_name", doctor_name)
                if clinic_name:
                    span.set_attribute("query.clinic_name", clinic_name)
                
                match_response = await self._client.match_doctor_hybrid(
                    query=doctor_name,
                    clinic_id=str(clinic_id) if clinic_id is not None else None,
                    clinic_name=None,
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
                    # No match found in the (possibly resolved) clinic. If we constrained by clinic_id,
                    # do a fallback global search to detect "doctor exists but in a different clinic".
                    if clinic_id is not None and doctor_name:
                        try:
                            fallback = await self._client.match_doctor_hybrid(
                                query=doctor_name,
                                clinic_id=None,
                                clinic_name=None,
                                top_k=5,
                                min_score_multi=0.6,
                                min_score_single=0.55,
                            )
                        except Exception:
                            fallback = None

                        if fallback and fallback.status == HybridMatchStatus.UNAMBIGUOUS_MATCH and fallback.best_match:
                            best = fallback.best_match
                            raise MCPWorkflowError(
                                (
                                    f"الاسم موجود، لكن مش في العيادة اللي اتذكرت. "
                                    f"الدكتور {best.name_ar or best.name_en} موجود في عيادة {best.clinic_name}. "
                                    "هل تحب أجيب مواعيده في العيادة دي؟"
                                ),
                                reason="provider_clinic_mismatch",
                                data={
                                    "requested_clinic": clinic_name,
                                    "candidates": [
                                        {
                                            "provider_id": _safe_parse_int(best.provider_id),
                                            "clinic_id": _safe_parse_int(best.clinic_id),
                                            "clinic_name": best.clinic_name,
                                            "name_ar": best.name_ar,
                                            "name_en": best.name_en,
                                            "score": best.score,
                                        }
                                    ],
                                },
                            )

                    # No match found at all
                    raise MCPWorkflowError(
                        match_response.message
                        or "ملقتش دكتور بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
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

        # If we have no clinic_id but do have a clinic_name, try resolving again using the matcher.
        # (This covers flows without doctor_name, e.g. "مواعيد عيادة ...")
        if clinic_id is None and clinic_name:
            clinic_match = await self._client.match_clinic_hybrid(query=clinic_name, top_k=5, min_score=0.65)
            if clinic_match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH and clinic_match.best_match:
                clinic_id = _safe_parse_int(clinic_match.best_match.clinic_id)
                clinic_name = clinic_match.best_match.clinic_name or clinic_name
                try:
                    if hasattr(state, "entities") and hasattr(state.entities, "clinic"):
                        state.entities.clinic = clinic_name
                    if hasattr(state, "entities") and hasattr(state.entities, "clinic_id"):
                        state.entities.clinic_id = clinic_id
                except Exception:
                    pass
                if provider_entry is None:
                    provider_entry = ProviderRecord(
                        clinic_id=clinic_id,
                        clinic_name_ar=clinic_name,
                        clinic_name_en=clinic_name,
                    )
            else:
                raise MCPWorkflowError(
                    clinic_match.message or "ملقتش عيادة بالاسم اللي اتذكر. ممكن تكتب الاسم بالكامل؟",
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
    # Capture sequences after "دكتور" / "د." / "د " (as a standalone token).
    # Important: guard against false positives inside other words, e.g. "مواعيد <X>"
    # contains the character "د" before a space, which previously matched "د\s+".
    doctor_patterns = [
        r"(?:^|\s)(?:دكتور|د\.?|د)\s+([\u0621-\u064A\w]+(?:\s+[\u0621-\u064A\w]+){0,2})",
    ]
    for pattern in doctor_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            candidate = _trim_punctuation(candidate)
            candidate = _strip_leading_doctor_words(candidate)
            candidate = _strip_trailing_clinic_words(candidate)
            if candidate and _is_plausible_doctor_name(candidate):
                return candidate
    return None


def _is_plausible_doctor_name(candidate: str) -> bool:
    """
    Guard against false positives like:
      - "مين دكتور النهارده متاح..." (not a name)
      - "دكتور مواعيد ..." (not a name)
    """
    if not candidate:
        return False
    tokens = [t for t in candidate.split() if t]
    if not tokens:
        return False

    # Disallow common non-name prefixes used in availability questions
    non_name_prefixes = {
        "مين",
        "من",
        "النهارده",
        "اليوم",
        "بكره",
        "امبارح",
        "متاح",
        "موجود",
        "مواعيد",
        "موعد",
        "بعد",
        "قبل",
        "عند",
        "في",
        "فى",
        "على",
        "لحد",
        "لغاية",
        "الساعة",
        "الساعه",
        "مساءا",
        "مساءً",
        "صباحا",
        "صباحًا",
    }
    if tokens[0].casefold() in non_name_prefixes:
        return False

    # If it's a single token and looks like a temporal word, reject.
    if len(tokens) == 1 and tokens[0].casefold() in non_name_prefixes:
        return False

    # Reject if all tokens are non-name words (e.g., "النهارده متاح")
    if all(t.casefold() in non_name_prefixes for t in tokens):
        return False

    # Require at least 2 tokens OR a token that is reasonably name-like (Arabic letters).
    # This reduces chance of "النهارده" being treated as a name.
    has_arabic = any(re.search(r"[\u0621-\u064A]", t) for t in tokens)
    if not has_arabic:
        return False
    if len(tokens) < 2:
        return False

    return True


def _detect_requested_intents(question: str, *, primary_intent: str, max_intents: int = 3) -> List[str]:
    """
    Detect multiple intents in a single user query (generic 2-3 combined asks),
    e.g. "مواعيد + سعر" or "مين دكتور + مواعيد + سعر".
    """
    q = (question or "").casefold()

    def has_any(patterns: List[str]) -> bool:
        return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)

    patterns = {
        "who_is_present": [
            r"مين\s+(?:اللي|اللى)?\s*(?:موجود(?:ين)?|متاح)",
            r"مين\s+(?:اللي|اللى)?\s*موجود\s+النهارده",
            r"مين\s+(?:اللي|اللى)?\s*موجود\s+بكره",
            r"مين\s+(?:اللي|اللى)?\s*موجود\s+بكرة",
        ],
        "ask_price": [
            r"\bسعر\b",
            r"\bكشف\b",
            r"\bالكشف\b",
            r"\bبكام\b",
            r"\bتكلف(?:ة|ه)?\b",
            r"\bprice\b",
            r"\bcost\b",
            r"\bfee\b",
        ],
        "check_availability": [
            r"\bمواعيد\b",
            r"\bموعد\b",
            r"\bمتاح\b",
            r"\bموجود\b",
            r"\bالنهارده\b",
            r"\bاليوم\b",
            r"\bجدول\b",
            r"\bschedule\b",
            r"\bavailable\b",
        ],
        "list_doctors": [
            r"مين\s+دكتور",
            r"مين\s+الدكتور",
            r"\bالدكاترة\b",
            r"\bاسماء\s+الدكاترة\b",
            r"\bأسماء\s+الدكاترة\b",
            r"\bdoctors?\b",
        ],
        "book_appointment": [
            r"\bاحجز\b",
            r"\bحجز\b",
            r"\bbook\b",
            r"\bappointment\b",
        ],
    }

    canonical_order = ["who_is_present", "list_doctors", "check_availability", "ask_price", "book_appointment"]
    found = [i for i in canonical_order if has_any(patterns[i])]

    # who_is_present subsumes list_doctors + schedule; avoid redundant handler calls.
    if "who_is_present" in found:
        found = ["who_is_present"] + [i for i in found if i not in {"who_is_present", "list_doctors", "check_availability"}]

    # Ensure primary is included and first.
    if primary_intent and primary_intent not in found:
        found.insert(0, primary_intent)
    elif primary_intent and primary_intent in found:
        found = [primary_intent] + [i for i in found if i != primary_intent]

    # Dedup + cap (keep stable order)
    out: List[str] = []
    for i in found:
        if i not in out:
            out.append(i)
        if len(out) >= max_intents:
            break
    return out


def _is_who_is_present_query(question: str) -> bool:
    q = (question or "").casefold()
    return bool(re.search(r"مين\s+(?:اللي|اللى)?\s*(?:موجود(?:ين)?|متاح)", q))


_AR_WEEKDAY_TO_EN = {
    "السبت": "saturday",
    "سبت": "saturday",
    "الأحد": "sunday",
    "الاحد": "sunday",
    "احد": "sunday",
    "الاثنين": "monday",
    "الاتنين": "monday",
    "اثنين": "monday",
    "الثلاثاء": "tuesday",
    "التلات": "tuesday",
    "ثلاثاء": "tuesday",
    "الأربعاء": "wednesday",
    "الاربعاء": "wednesday",
    "الأربع": "wednesday",
    "الاربع": "wednesday",
    "الخميس": "thursday",
    "خميس": "thursday",
    "الجمعة": "friday",
    "جمعه": "friday",
    "جمعه": "friday",
}


def _infer_target_day_id(question: str, *, now_dt: datetime | None) -> int:
    """
    Infer MCP day_id (1=Saturday..7=Friday) from Arabic relative day expressions.
    Defaults to "today" if not specified.
    """
    q = (question or "").casefold()

    base_dt = now_dt or datetime.now()

    if any(tok in q for tok in ["بكره", "بكرة", "غدا", "غداً"]):
        target = base_dt + timedelta(days=1)
        return DAY_NAME_TO_ID[target.strftime("%A").lower()]

    if any(tok in q for tok in ["النهارده", "نهارده", "اليوم"]):
        return DAY_NAME_TO_ID[base_dt.strftime("%A").lower()]

    for ar, en in _AR_WEEKDAY_TO_EN.items():
        if ar in q:
            return DAY_NAME_TO_ID[en]

    # Fallback: treat as today
    return DAY_NAME_TO_ID[base_dt.strftime("%A").lower()]


def _filter_provider_list_by_clinic_id(provider_list: ProviderListPayload, clinic_id: int) -> ProviderListPayload:
    filtered = [p for p in provider_list.providers if p.clinic_id == clinic_id]
    return ProviderListPayload(providers=filtered)


def _format_who_is_present(*, clinic_label: str, day_id: int, present: list[tuple[str, list[str]]]) -> str:
    lines = [
        "قائمة الموجودين حسب بيانات السيستم.",
        f"العيادة: {clinic_label}.",
        f"اليوم (day_id): {day_id}.",
        "\nالموجودين:",
    ]
    for name, times in present:
        if times:
            lines.append(f"- {name}: " + "، ".join(times))
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


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
            # Guard: "عيادة دكتور <name>" is NOT a clinic name; user likely means "the doctor's clinic".
            head = (candidate.split()[0] if candidate.split() else "").casefold().strip(".")
            if head in {"دكتور", "د", "dr", "doctor"}:
                continue
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

    def _normalize_ampm_for_ar(value: str) -> str:
        """Prefer صباحًا/مساءً over ص/م or AM/PM in schedule display strings."""
        if not value:
            return value
        out = value
        out = re.sub(r"(?i)\bA\.?M\.?\b", "صباحًا", out)
        out = re.sub(r"(?i)\bP\.?M\.?\b", "مساءً", out)
        out = re.sub(r"(\d{1,2}[:.]\d{2})\s*ص\b", r"\1 صباحًا", out)
        out = re.sub(r"(\d{1,2}[:.]\d{2})\s*م\b", r"\1 مساءً", out)
        out = re.sub(r"صباح(?:ا|اً|ً|ًا)", "صباحًا", out)
        out = re.sub(r"مساء(?:ا|اً|ً|ًا)", "مساءً", out)
        return out

    grouped: Dict[str, List[str]] = defaultdict(list)
    for slot in response.slots:
        day = slot.day_name or f"اليوم #{slot.day_id}"
        start = _normalize_ampm_for_ar(slot.shift_start or "غير محدد")
        end = _normalize_ampm_for_ar(slot.shift_end or "غير محدد")
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

