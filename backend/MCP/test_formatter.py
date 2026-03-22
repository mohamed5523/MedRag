import json
from clinic_server import _format_mcp_response

data = {
  "success": True,
  "data": [
    {
      "clinicId": 1,
      "clinicName": "عيادة الجراحة",
      "doctors": [
        {"providerId": 101, "DoctorNameA": "د. أحمد", "availability": "10:00 - 14:00"},
        {"providerId": 102, "DoctorNameA": "د. محمد", "availability": "14:00 - 20:00"}
      ]
    }
  ]
}

print(_format_mcp_response(data, "Test"))
