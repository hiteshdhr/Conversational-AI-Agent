from typing import TypedDict, Annotated, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage

class UserDetailsDict(TypedDict):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intent: Optional[str]
    user_details: UserDetailsDict
    tool_triggered: bool
    in_funnel: bool   # sticky flag - stays True once high intent detected
