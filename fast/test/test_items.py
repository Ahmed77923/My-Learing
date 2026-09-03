from fastapi.testclient import TestClient
from fast.main import app

client = TestClient(app)


# =========================
# GET /
# =========================

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to my FastAPI application"}


# =========================
# POST /items/
# =========================

def test_create_item():
    response = client.post(
        "/items/",
        json={
            "name": "Laptop",
            "quantity": 10,
            "price": 1000,
            "category": "Electronics",
            "expiration": None
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 2
    assert data["name"] == "Laptop"
    assert data["quantity"] == 10
    assert data["price"] == 1000
    assert data["category"] == "Electronics"


# =========================
# GET /items/{item_id}
# =========================

def test_get_item():
    response = client.get("/items/1")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert data["name"] == "Item 1"
    assert data["quantity"] == 10


def test_get_item_not_found():
    response = client.get("/items/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"


# =========================
# PUT /items/{item_id}
# =========================

def test_update_item():
    response = client.put(
        "/items/1",
        json={
            "name": "Updated Item",
            "quantity": 50,
            "price": 20.5,
            "category": "Updated Category",
            "expiration": None
        }
    )

    assert response.status_code == 200

    data = response.json()

    # ID must remain 1
    assert data["id"] == 1

    assert data["name"] == "Updated Item"
    assert data["quantity"] == 50
    assert data["price"] == 20.5
    assert data["category"] == "Updated Category"


def test_update_item_not_found():
    response = client.put(
        "/items/9999",
        json={
            "name": "Test",
            "quantity": 10,
            "price": 10,
            "category": "Test",
            "expiration": None
        }
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"


# =========================
# PATCH /items/{item_id}/stock
# =========================

def test_update_stock():
    response = client.patch(
        "/items/1/stock",
        json={
            "quantity": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    # 50 + 5 = 55
    assert data["quantity"] == 55


def test_update_stock_zero():
    response = client.patch(
        "/items/1/stock",
        json={
            "quantity": 0
        }
    )

    assert response.status_code == 400


def test_update_stock_negative():
    response = client.patch(
        "/items/1/stock",
        json={
            "quantity": -5
        }
    )

    assert response.status_code == 400


def test_update_stock_item_not_found():
    response = client.patch(
        "/items/9999/stock",
        json={
            "quantity": 5
        }
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"


# =========================
# DELETE /items/{item_id}
# =========================

def test_delete_item():
    # Create an item first
    create_response = client.post(
        "/items/",
        json={
            "name": "Delete Me",
            "quantity": 5,
            "price": 10,
            "category": "Test",
            "expiration": None
        }
    )

    assert create_response.status_code == 200

    item_id = create_response.json()["id"]

    # Delete it
    response = client.delete(f"/items/{item_id}")

    assert response.status_code == 200
    assert response.json()["message"] == "Item deleted successfully"

    # Verify it no longer exists
    response = client.get(f"/items/{item_id}")

    assert response.status_code == 404


def test_delete_item_not_found():
    response = client.delete("/items/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"