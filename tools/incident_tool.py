from mcp_sdk.tool import BaseTool
import requests

class IncidentTool(BaseTool):
    name = "incident_lookup"
    description = "Look up on-call incidents from incident repo/API"
    # Incident Knowledge Integration

    def run(self, incident_id: str):
        resp = requests.get(f"https://oncall.mycorp.com/api/incidents/{incident_id}")
        return resp.json()
