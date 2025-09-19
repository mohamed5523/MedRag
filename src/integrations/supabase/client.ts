import { createClient } from '@supabase/supabase-js';
import type { Database } from './types';

// Prefer env vars; fall back to previous defaults so the app doesn't white screen
const FALLBACK_URL = 'https://uwxqpxyfxqhwlqzniyyh.supabase.co';
const FALLBACK_ANON = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV3eHFweHlmeHFod2xxem5peXloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgxMTgwMDEsImV4cCI6MjA3MzY5NDAwMX0.qr30ISF5I_TxJb2GwuQ-Nt8MaoLxnPzMzA8M8g-l4_Q';

const SUPABASE_URL = (import.meta.env.VITE_SUPABASE_URL as string) || FALLBACK_URL;
const SUPABASE_PUBLISHABLE_KEY = (import.meta.env.VITE_SUPABASE_ANON_KEY as string) || FALLBACK_ANON;

if (!import.meta.env.VITE_SUPABASE_URL || !import.meta.env.VITE_SUPABASE_ANON_KEY) {
  // eslint-disable-next-line no-console
  console.warn('Using fallback Supabase credentials. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to use your own project.');
}

export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
  auth: {
    storage: localStorage,
    persistSession: true,
    autoRefreshToken: true,
  }
});