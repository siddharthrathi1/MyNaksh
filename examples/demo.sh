#!/usr/bin/env bash
# Replays the assignment's example conversation against a running API.
# Usage: ./examples/demo.sh [base_url]     (default http://localhost:8000)
set -euo pipefail

BASE="${1:-http://localhost:8000}"
USER_ID="rahul-$(date +%s)"   # fresh user each run

post() {
  local session="$1" message="$2"
  echo -e "\n>>> [$session] $message"
  curl -s -X POST "$BASE/chat" -H 'content-type: application/json' \
    -d "{\"user_id\":\"$USER_ID\",\"session_id\":\"$USER_ID-$session\",\"message\":\"$message\"}" \
    | python3 -m json.tool
}

# First conversation: disclosure → long-term memory
post s1 "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."
# Same conversation: personalised answer from the Shared Brain
post s1 "What should I focus on for my career?"
# Follow-up: resolved from short-term context
post s1 "Why do you say that?"
# New conversation: long-term recall, no old history
post s2 "What do you remember about my career goals?"
# Irrelevant memory must not leak into an unrelated topic
post s2 "How will my love life be this year?"
# Correction
post s2 "Actually I was born in Mumbai, not Delhi."

echo -e "\n>>> Profile"
curl -s "$BASE/users/$USER_ID" | python3 -m json.tool
echo -e "\n>>> Shared Brain (including superseded/retracted)"
curl -s "$BASE/users/$USER_ID/memories?include_inactive=true" | python3 -m json.tool
