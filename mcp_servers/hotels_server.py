
from typing import List, Optional
from pydantic import BaseModel, Field
from fastmcp import MCPApp
from ..domain.hotels import search_hotels

app = MCPApp(name="HotelsAgent", version="0.1.0", description="Hotel search & details")

class HotelsSearchInput(BaseModel):
    destination: str
    check_in: str
    check_out: str
    guests: int = Field(ge=1)
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    stars: Optional[List[int]] = None

class HotelBrief(BaseModel):
    name: str
    stars: int
    price_per_night: float
    location: str
    free_cancellation: bool
    url: Optional[str] = None
    hotel_id: Optional[str] = None

class HotelsSearchOutput(BaseModel):
    results: List[HotelBrief]

@app.tool(
    name="hotels.search",
    desc="Search hotels with basic filters.",
    input_model=HotelsSearchInput,
    output_model=HotelsSearchOutput,
)
def hotels_search(inp: HotelsSearchInput) -> HotelsSearchOutput:
    data = search_hotels(
        destination=inp.destination,
        check_in=inp.check_in,
        check_out=inp.check_out,
        guests=inp.guests,
        budget_min=inp.budget_min,
        budget_max=inp.budget_max,
        stars=inp.stars,
    )
    return HotelsSearchOutput(results=[HotelBrief(**d) for d in data])

if __name__ == "__main__":
    app.run()
