import requests

ACCESS_TOKEN = "dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA="
conversation_id = "215469480599086"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

response = requests.get(
    f"https://api.intercom.io/conversations/{conversation_id}",
    headers=headers
)

if response.status_code == 200:
    data = response.json()

    # Print all messages
    print("🟡 Source Message:")
    print(data['source']['body'])

    print("\n🟢 Conversation Replies:")
    for part in data.get('conversation_parts', {}).get('conversation_parts', []):
        author_type = part['author']['type']
        message = part['body']
        print(f"\n[{author_type}] {message}")
else:
    print("Error:", response.status_code, response.text)
