
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Hotel:
    name: str
    stars: int
    price_per_night: float
    location: str
    free_cancellation: bool
    url: Optional[str] = None
    hotel_id: Optional[str] = None

def search_hotels(destination: str, check_in: str, check_out: str, guests: int,
                  budget_min: Optional[int]=None, budget_max: Optional[int]=None,
                  stars: Optional[List[int]]=None) -> List[Dict[str, Any]]:
    # TODO: integrate Booking/Amadeus. Stubbed.
    mock = [
        Hotel("Kyoto Garden Inn", 3, 120.0, "Central Kyoto", True, "https://example.com/h/gardeninn", "h_kyoto_garden"),
        Hotel("The Royal Kyoto", 5, 310.0, "Gion District", False, "https://example.com/h/royalkyoto", "h_royal_kyoto"),
    ]
    out = []
    for h in mock:
        if budget_min is not None and h.price_per_night < budget_min:
            continue
        if budget_max is not None and h.price_per_night > budget_max:
            continue
        if stars and h.stars not in stars:
            continue
        out.append(asdict(h))
    return out
