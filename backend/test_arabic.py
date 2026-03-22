import os
import sys

# Add backend to path
sys.path.append("/home/morad/Projects/heal-query-hub/backend")

from app.core.qa_engine import QAEngine

qa = QAEngine()
test_str = "مواعيد دكتور ميلاد عبده"
res = qa._is_arabic_query(test_str)
print(f"Result for '{test_str}': {res}")

from app.core.intent_router import get_router
from app.core.state_manager import ConversationState, Entities

state = ConversationState(
    entities=Entities(doctor="ميلاد عبده", clinic=None, hospital=None),
    intent="unknown",
    target_entity_type="doctor",
    last_user_question=test_str,
    needs_followup=False
)

router = get_router()
route = router.decide_route(state, test_str)
print(f"Route Decision: {route.mode.value}, {route.intent}")
