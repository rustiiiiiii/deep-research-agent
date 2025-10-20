
from typing import Optional, List
from pydantic import BaseModel, Field
from fastmcp import MCPApp
from ..domain.flights import search_flights

app = MCPApp(name="FlightsAgent", version="0.1.0", description="Flight search")

class FlightsSearchInput(BaseModel):
    origin: str
    destination: str
    depart: str
    return_: Optional[str] = Field(None, alias="return")
    pax: int = Field(ge=1)
    cabin: Optional[str] = "economy"

class FlightOption(BaseModel):
    carrier: str
    flight_no: str
    depart_time: str
    arrive_time: str
    duration_min: int
    stops: int
    fare_total: float
    url: Optional[str] = None

class FlightsSearchOutput(BaseModel):
    options: List[FlightOption]

@app.tool(
    name="flights.search",
    desc="Search flights for given dates and pax.",
    input_model=FlightsSearchInput,
    output_model=FlightsSearchOutput,
)
def flights_search(inp: FlightsSearchInput) -> FlightsSearchOutput:
    data = search_flights(
        origin=inp.origin,
        destination=inp.destination,
        depart=inp.depart,
        return_=inp.return_,
        pax=inp.pax,
        cabin=inp.cabin,
    )
    return FlightsSearchOutput(options=[FlightOption(**d) for d in data])

if __name__ == "__main__":
    app.run()
