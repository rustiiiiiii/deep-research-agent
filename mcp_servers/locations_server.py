
from typing import Optional, List
from pydantic import BaseModel, Field
from fastmcp import MCPApp
from ..domain.locations import search_places

app = MCPApp(name="LocationsAgent", version="0.1.0", description="Attractions & POIs")

class PlacesSearchInput(BaseModel):
    destination: str
    themes: Optional[List[str]] = Field(None, description="culture|food|nature|shopping")
    open_now: Optional[bool] = None
    with_kids: Optional[bool] = None

class PlaceBrief(BaseModel):
    name: str
    category: str
    highlights: List[str]
    hours: Optional[str] = None
    tickets: Optional[str] = None
    url: Optional[str] = None
    place_id: Optional[str] = None

class PlacesSearchOutput(BaseModel):
    results: List[PlaceBrief]

@app.tool(
    name="places.search",
    desc="Search notable places/attractions.",
    input_model=PlacesSearchInput,
    output_model=PlacesSearchOutput,
)
def places_search(inp: PlacesSearchInput) -> PlacesSearchOutput:
    data = search_places(
        destination=inp.destination,
        themes=inp.themes,
        open_now=inp.open_now,
        with_kids=inp.with_kids,
    )
    return PlacesSearchOutput(results=[PlaceBrief(**d) for d in data])

if __name__ == "__main__":
    app.run()
