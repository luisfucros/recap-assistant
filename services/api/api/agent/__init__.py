"""The LangGraph reading assistant: nodes, tools, guardrails, and schemas.

The agent runs a plan→act→observe loop over a LangChain chat model. Its internal
nodes emit schema-validated structured output (``agent.schemas``); guardrails
screen input and output (``agent.guardrails``). The three user-scoped read tools
(``agent.tools``) wrap the reading services and close over a per-turn
``agent.context.ToolContext`` so the owner is injected server-side, never taken
from tool arguments.

``agent.graph`` assembles these into the compiled state graph
(``normalize_input → guardrail_in → load_progress → plan → generate ⇄ tools →
persist_memory → guardrail_out``), threading one ``agent.state.AgentState`` per
turn. ``api.services.agent_service.AgentService`` runs or streams a turn over it,
turning it into the ordered ``agent.events`` (tool steps, then answer tokens, then
``done`` — or a lone ``blocked``). The ``/chat`` HTTP surface and the durable
checkpointer land in the next slice.
"""
