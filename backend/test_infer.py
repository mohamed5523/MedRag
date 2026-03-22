from app.services.clinic_workflow import _infer_clinic_from_context
from app.core.models import ConversationState

def test():
    state = ConversationState()
    queries = [
        "مين دكتور أطفال متاح حاليا",
        "فين دكتور أسنان",
        "عايز دكتور عظام النهارده",
        "محتاج دكتور باطنة"
    ]
    for q in queries:
        inferred = _infer_clinic_from_context(state, q)
        print(f"Query: {q:40} | Inferred Clinic: {inferred}")

if __name__ == "__main__":
    test()
