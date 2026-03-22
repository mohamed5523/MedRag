from typing import Any

def _format_mcp_response(data: Any, title: str = "Data") -> str:
    if not data:
        return "No data available."
    if isinstance(data, dict):
        if "data" in data and len(data) <= 2:
            return _format_mcp_response(data["data"], title)
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"### {k}")
                lines.append(_format_mcp_response(v, k))
            else:
                lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)
    elif isinstance(data, list):
        if not data:
            return "Empty list."
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                name_keys = ["DoctorNameA", "DoctorNameL", "clinicName", "serviceNameA", "ServiceName", "name", "title"]
                item_title = f"Item {i+1}"
                for nk in name_keys:
                    if nk in item and item[nk]:
                        item_title = str(item[nk])
                        break
                lines.append(f"\n#### {item_title}")
                for k, v in item.items():
                    if k in name_keys and str(v) == item_title: continue
                    if v is None or v == "" or v == [] or v == {}: continue
                    if isinstance(v, (dict, list)):
                        lines.append(f"  - **{k}**:")
                        sub_lines = _format_mcp_response(v, k).split("\n")
                        lines.extend([f"    {sl}" for sl in sub_lines if sl.strip()])
                    else:
                        lines.append(f"  - **{k}**: {v}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)
    return str(data)

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
