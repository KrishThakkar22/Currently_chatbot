import requests
import json
import time

HEADERS = {
    "Authorization": "Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",
    "Accept": "application/json"
}

BASE_URL = "https://api.intercom.io/conversations"
OUTPUT_FILE = "intercom_conversations.json"


def fetch_conversations():
    all_conversations = []
    url = BASE_URL

    while url:
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()

            # Append the current batch of conversations
            conversations = data.get("conversations", [])
            all_conversations.extend(conversations)

            print(f"Fetched {len(conversations)} conversations, total: {len(all_conversations)}")

            # Save to file after each batch
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_conversations, f, ensure_ascii=False, indent=2)

            # Prepare for next page
            next_page = data.get("pages", {}).get("next", None)
            if next_page and "starting_after" in next_page:
                url = f"{BASE_URL}?starting_after={next_page['starting_after']}"
                time.sleep(0.5)  # Avoid rate-limiting
            else:
                url = None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

    print(f"Finished fetching. Total conversations saved: {len(all_conversations)}")


if __name__ == "__main__":
    fetch_conversations()
