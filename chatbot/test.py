import httpx

# INTERCOM_ACCESS_TOKEN = "your_admin_token_here"

async def fetch_intercom_admins():
    url = "https://api.intercom.io/admins"
    headers = {
        "Authorization": f"Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        print("🧑‍💼 Intercom Admins:")
        for admin in data.get("admins", []):
            print(f"👤 Name: {admin['name']}")
            print(f"📧 Email: {admin['email']}")
            print(f"🆔 Admin ID: {admin['id']}\n")

        return data.get("admins", [])

# Example usage
import asyncio
asyncio.run(fetch_intercom_admins())
