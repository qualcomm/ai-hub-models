#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Fetch a JIRA issue and output its fields as JSON.
#
# Usage: fetch_jira_issue.sh <jira_token> <ticket>
#
# Outputs (stdout, JSON object):
#   issue_type       - e.g. "Onboarding", "Bug", "Task"
#   type_of_request  - value of "Type of request" custom field (or "")
#   labels           - comma-separated list of JIRA labels
#   status_name      - e.g. "In Progress", "Closed"
#   status_category  - e.g. "new", "indeterminate", "done"

set -euo pipefail

JIRA_TOKEN="$1"
TICKET="$2"
JIRA_BASE_URL="${JIRA_BASE_URL:-https://jira-dc-tools.qualcomm.com/jira/rest/api/2}"

RESPONSE=$(curl -sS -w "\n%{http_code}" \
  -H "Authorization: Bearer ${JIRA_TOKEN}" \
  -H "Content-Type: application/json" \
  "${JIRA_BASE_URL}/issue/${TICKET}?fields=status,issuetype,labels,customfield_32716")

HTTP_STATUS=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_STATUS" = "404" ]; then
  echo "JIRA ticket '${TICKET}' does not exist." >&2
  jq -n '{error: "not_found", message: "Ticket does not exist"}'
elif [ "$HTTP_STATUS" != "200" ]; then
  echo "Failed to fetch JIRA ticket '${TICKET}' (HTTP $HTTP_STATUS)." >&2
  jq -n --arg status "$HTTP_STATUS" '{error: "request_failed", message: ("HTTP " + $status)}'
else
  echo "$BODY" | jq '{
    issue_type: .fields.issuetype.name,
    type_of_request: (.fields.customfield_32716.value // ""),
    labels: ([.fields.labels[]] | join(",")),
    status_name: .fields.status.name,
    status_category: .fields.status.statusCategory.key
  }'
fi
