from datetime import date
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="My FastAPI Application",
    description="This is a sample FastAPI application.",
    version="1.0.0",
)


# -------------------------
# Request Models
# -------------------------

class StockUpdate(BaseModel):
    quantity: int


class ItemCreate(BaseModel):
    name: str
    quantity: int = 0
    price: float
    category: str
    expiration: Optional[date] = None


# -------------------------
# Full Item Model
# -------------------------

class Item(ItemCreate):
    id: int


# -------------------------
# In-memory database
# -------------------------

items: List[Item] = [
    Item(
        id=1,
        name="Item 1",
        quantity=10,
        price=9.99,
        category="Category A",
        expiration=None,
    )
]


# -------------------------
# Root
# -------------------------

@app.get("/")
async def root():
    return {
        "message": "Welcome to my FastAPI application"
    }


# -------------------------
# Create Item
# -------------------------

@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):

    # Generate a new ID
    new_id = max(
        [existing_item.id for existing_item in items],
        default=0
    ) + 1

    # Create full Item with server-generated ID
    new_item = Item(
        id=new_id,
        **item.model_dump()
    )

    items.append(new_item)

    return new_item


# -------------------------
# Get Item
# -------------------------

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):

    for item in items:

        if item.id == item_id:
            return item

    raise HTTPException(
        status_code=404,
        detail="Item not found"
    )


# -------------------------
# Update Item
# -------------------------

@app.put("/items/{item_id}", response_model=Item)
async def update_item(
    item_id: int,
    updated_item: ItemCreate
):

    for index, item in enumerate(items):

        if item.id == item_id:

            # Keep the original ID
            new_item = Item(
                id=item_id,
                **updated_item.model_dump()
            )

            items[index] = new_item

            return new_item

    raise HTTPException(
        status_code=404,
        detail="Item not found"
    )


# -------------------------
# Update Stock
# -------------------------

@app.patch("/items/{item_id}/stock", response_model=Item)
async def update_stock(
    item_id: int,
    stock_update: StockUpdate
):

    if stock_update.quantity <= 0:
        raise HTTPException(
            status_code=400,
            detail="Quantity must be greater than 0"
        )

    for item in items:

        if item.id == item_id:

            item.quantity += stock_update.quantity

            return item

    raise HTTPException(
        status_code=404,
        detail="Item not found"
    )


# -------------------------
# Delete Item
# -------------------------

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):

    for item in items:

        if item.id == item_id:

            items.remove(item)

            return {
                "message": "Item deleted successfully"
            }

    raise HTTPException(
        status_code=404,
        detail="Item not found"
    )

