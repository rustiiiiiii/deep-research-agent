
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Flight:
    carrier: str
    flight_no: str
    depart_time: str
    arrive_time: str
    duration_min: int
    stops: int
    fare_total: float
    url: Optional[str] = None

def search_flights(origin: str, destination: str, depart: str, return_: Optional[str],
                   pax: int, cabin: Optional[str]="economy") -> List[Dict[str, Any]]:
    # TODO: integrate Amadeus/Skyscanner. Stubbed.
    mock = [
        Flight("ANA", "NH830", f"{depart}T10:30", f"{depart}T19:15", 525, 1, 640.0, "https://example.com/f/nh830"),
    ]
    return [asdict(x) for x in mock]
