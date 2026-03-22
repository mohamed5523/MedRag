import asyncio
from app.core.state_manager import StateManager
from app.services.clinic_workflow import _infer_clinic_from_context

async def run():
    sm = StateManager()
    queries = [
        "مين دكتور أطفال متاح حاليا",
        "فين دكتور أسنان",
        "عايز دكتور عظام النهارده",
        "محتاج دكتور باطنة"
    ]
    for q in queries:
        state = await sm.extract_state(q, "test_session")
        print(f"\nQuery: {q}")
        print(f"LLM Intent: {state.intent}")
        print(f"LLM Target: {state.target_entity_type}")
        print(f"LLM Clinic: {state.entities.clinic}")
        print(f"LLM Doctor: {state.entities.doctor}")
        
        inferred = _infer_clinic_from_context(state, q)
        print(f"Fallback Inferred Clinic: {inferred}")

if __name__ == "__main__":
    asyncio.run(run())
