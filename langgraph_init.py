from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tools import get_weather, make_answer_about_job_description



def init_agent(model):
    job_tool = make_answer_about_job_description(model)

    agent = create_react_agent(
        # model="anthropic:claude-3-7-sonnet-latest",
        model=model,
        tools=[get_weather, job_tool],
        prompt="You are a helpful assistant"
    )
    return agent