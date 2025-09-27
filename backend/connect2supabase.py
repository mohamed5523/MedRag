import logging
import os

try:
    from supabase import create_client
except Exception:  # pragma: no cover - import guard for environments without supabase
    create_client = None  # type: ignore

logger = logging.getLogger(__name__)


class _MissingSupabase:
    """Proxy that raises a clear error when Supabase isn't configured."""

    def __getattr__(self, _name):  # noqa: D401
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) in backend .env"
        )


def _init_supabase_client():
    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    # Prefer service role for server-side actions (deletes), fall back to anon if present
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        logger.warning(
            "Supabase env vars missing. SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/ANON_KEY are required for storage/db operations."
        )
        return _MissingSupabase()

    if create_client is None:
        logger.warning("supabase python client not installed; endpoints using Supabase will fail")
        return _MissingSupabase()

    try:
        client = create_client(url, key)
        logger.info("Supabase client initialized")
        return client
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to initialize Supabase client: {exc}")
        return _MissingSupabase()


# Exported client used by API modules
supabase = _init_supabase_client()


