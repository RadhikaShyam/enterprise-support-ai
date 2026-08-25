import requests


BASE_URL = "http://127.0.0.1:8000"

REGISTER_URL = f"{BASE_URL}/auth/register"
LOGIN_URL = f"{BASE_URL}/auth/login"
UPLOAD_URL = f"{BASE_URL}/documents/upload"


USER_A = {
    "email": "isolation-user-a@example.com",
    "password": "TestPassword123!",
}


def test_connection():

    print("\n--- TESTING API ---")

    response = requests.get(
        f"{BASE_URL}/docs",
        timeout=10,
    )

    print("DOCS:", response.status_code)

    print("\n--- REGISTER ---")

    response = requests.post(
        REGISTER_URL,
        json=USER_A,
        timeout=10,
    )

    print(
        "REGISTER:",
        response.status_code,
        response.text,
    )

    print("\n--- LOGIN ---")

    response = requests.post(
        LOGIN_URL,
        json=USER_A,
        timeout=10,
    )

    print(
        "LOGIN:",
        response.status_code,
        response.text,
    )

    assert response.status_code == 200

    token = response.json()["access_token"]

    print("\nTOKEN RECEIVED")

    print("\n--- UPLOAD ---")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    with open(
        "data/uploads/password-support.txt",
        "rb",
    ) as file:

        response = requests.post(
            UPLOAD_URL,
            headers=headers,
            files={
                "file": (
                    "password-support.txt",
                    file,
                    "text/plain",
                )
            },
            timeout=120,
        )

    print(
        "UPLOAD:",
        response.status_code,
        response.text,
    )

    assert response.status_code == 200