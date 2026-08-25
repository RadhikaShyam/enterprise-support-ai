import requests


BASE_URL = "http://127.0.0.1:8000"

REGISTER_URL = f"{BASE_URL}/auth/register"
LOGIN_URL = f"{BASE_URL}/auth/login"
UPLOAD_URL = f"{BASE_URL}/documents/upload"
RAG_URL = f"{BASE_URL}/documents/rag"


TEST_EMAIL = "rag-test@example.com"
TEST_PASSWORD = "TestPassword123!"


TEST_CASES = [
    {
        "id": "rag-001",
        "question": "What happens after five unsuccessful login attempts?",
        "expected": [
            "temporarily locked",
            "30 minutes",
        ],
    },
    {
        "id": "rag-002",
        "question": "Where can employees reset their corporate password?",
        "expected": [
            "Enterprise Identity Portal",
        ],
    },
    {
        "id": "rag-003",
        "question": "What are the password requirements?",
        "expected": [
            "12 characters",
            "uppercase",
            "lowercase",
            "number",
            "special character",
        ],
    },
    {
        "id": "rag-004",
        "question": "What should employees do if MFA continues to fail?",
        "expected": [
            "Enterprise IT Service Desk",
        ],
    },
    {
        "id": "rag-005",
        "question": "What are the company's vacation benefits?",
        "expected": [],
    },
    {
        "id": "rag-006",
        "question": "What is the company's maternity leave policy?",
        "expected": [],
    },
    {
        "id": "rag-007",
        "question": "What is the company's stock price?",
        "expected": [],
    },
    {
        "id": "rag-008",
        "question": "What are the IT Service Desk support hours?",
        "expected": [
            "Monday through Friday",
            "9:00 AM",
            "6:00 PM",
        ],
    },
]


FALLBACK = (
    "I don't have enough information in the available "
    "support documentation to answer that question."
)


def get_token():

    register_response = requests.post(
        REGISTER_URL,
        json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
        },
        timeout=30,
    )

    # 200/201 means newly created.
    # 400 means the user probably already exists.
    if register_response.status_code not in (200, 201, 400):
        raise AssertionError(
            f"Registration failed: "
            f"{register_response.status_code} "
            f"{register_response.text}"
        )

    login_response = requests.post(
        LOGIN_URL,
        json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
        },
        timeout=30,
    )

    assert login_response.status_code == 200, (
        f"Login failed: "
        f"{login_response.status_code}: "
        f"{login_response.text}"
    )

    return login_response.json()["access_token"]


def upload_test_document(token):

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

    assert response.status_code == 200, (
        f"Document upload failed: "
        f"{response.status_code}: "
        f"{response.text}"
    )

    data = response.json()

    assert data["status"] == "completed"

    return data


def test_rag_regression():

    token = get_token()

    upload_test_document(token)

    headers = {
        "Authorization": f"Bearer {token}"
    }

    failures = []

    for case in TEST_CASES:

        response = requests.post(
            RAG_URL,
            headers=headers,
            json={
                "question": case["question"],
            },
            timeout=120,
        )

        assert response.status_code == 200, (
            f"{case['id']} returned "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

        data = response.json()

        answer = data.get("answer", "")
        answer_lower = answer.lower()

        expected = case["expected"]

        if not expected:

            if answer != FALLBACK:
                failures.append(
                    f"{case['id']}: expected fallback, "
                    f"got: {answer}"
                )

            continue

        for expected_fact in expected:

            if expected_fact.lower() not in answer_lower:

                failures.append(
                    f"{case['id']}: missing "
                    f"'{expected_fact}' "
                    f"in answer: {answer}"
                )

    assert not failures, "\n".join(failures)