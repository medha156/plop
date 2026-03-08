gcloud alpha logging tail --format=json | jq -r '
  def color:
    if . == "ERROR" then "\u001b[31m"
    elif . == "WARNING" then "\u001b[33m"
    else "\u001b[32m" end;

  . as $log |
  ($log.severity | color) + "[" + ($log.severity // "INFO") + "]\u001b[0m " +
  (
    $log.textPayload
    // ($log.jsonPayload.message // ($log.jsonPayload | tostring))
    // ($log.protoPayload | tostring)
  )
'