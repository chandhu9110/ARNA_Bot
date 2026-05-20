from fastapi.testclient import TestClient
from pathlib import Path
from Backend.src.main import app

client = TestClient(app)

def test_analyze():

    image_path = Path("tests/image.JPG").resolve()

    assert image_path.exists()

    with open(image_path, "rb") as image:

        response = client.post(
            "/analyze",
            files={
                "file": ("image.JPG", image, "image/jpeg")
            }
        )

    print(response.json())

    assert response.status_code == 200