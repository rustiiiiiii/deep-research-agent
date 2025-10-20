
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Place:
    name: str
    category: str
    highlights: List[str]
    hours: Optional[str] = None
    tickets: Optional[str] = None
    url: Optional[str] = None
    place_id: Optional[str] = None

def search_places(destination: str, themes: Optional[List[str]]=None,
                  open_now: Optional[bool]=None, with_kids: Optional[bool]=None) -> List[Dict[str, Any]]:
    # TODO: integrate Google Places/TripAdvisor. Stubbed.
    mock = [
        Place("Fushimi Inari Taisha", "Shrine", ["Torii gates","Hiking trail","View"], "24/7", "Free", "https://example.com/p/fushimi", "p_001"),
        Place("Kiyomizu-dera", "Temple", ["UNESCO site","Wooden terrace","View"], "6am–6pm", "¥400", "https://example.com/p/kiyomizu", "p_002"),
    ]
    return [asdict(x) for x in mock]
