import asyncio
import logging
import sys

sys.path.append("/home/morad/Projects/heal-query-hub/backend")

from app.core.qa_engine import QAEngine
from app.core.intent_router import get_router
from app.core.state_manager import ConversationState, Entities
from app.services.clinic_workflow import ClinicWorkflowService

logging.basicConfig(level=logging.DEBUG)

async def test_workflow():
    qa = QAEngine()
    workflow = ClinicWorkflowService()
    router = get_router()
    
    query = "مواعيد دكتور ميلاد عبده"
    state = ConversationState(
        intent="check_availability",
        entities=Entities(doctor="ميلاد عبده"),
        target_entity_type="doctor",
        last_user_question=query,
        needs_followup=False,
    )
    
    decision = router.decide_route(state, query)
    print(f"Decision: mode={decision.mode.value}, intent={decision.intent}")
    
    try:
        result = await workflow.run(
            decision=decision,
            state=state,
            question=query,
            qa_engine=qa,
        )
        print("Workflow Success:")
        print(result.qa_response.get("answer"))
    except Exception as e:
        print(f"Workflow Exception: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_workflow())
