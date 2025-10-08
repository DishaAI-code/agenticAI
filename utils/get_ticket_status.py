# utils/ticket_status.py
import requests

TICKET_API_URL = "https://s22-api.nopaperforms.net/tickets/v1/details"
HEADERS = {
    "secret-key": "94d665300cdfade00ef428d0bc6ffcfd",
    "access-key": "31d6bc3c2bd64086a82c8f4467ea4a88",
    "Content-Type": "application/json",
    "Cookie": "PHPSESSID=cga0t5hs7uoblvksib5tone96r"
}

def get_ticket_status(ticket_id: str) -> str:
    """Fetch the status of a ticket via API"""
    payload = {"ticket_id": ticket_id}

    try:
        response = requests.post(TICKET_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()

        data = response.json()
        print(data)
        if data.get("status") is True:
            details = data["data"]["details"]
            description = details.get("ticket_description", "No description")
            status = details.get("status", "Unknown")
            return f"Your ticket '{description}' is currently '{status}'."
        else:
            return "Sorry, I couldn't fetch your ticket details at the moment."

    except Exception as e:
        print(f"Error fetching ticket: {e}")
        return "There was an error fetching your ticket status."


get_ticket_status("ec89b530e1c74e44be2f1f5b569f6c79")