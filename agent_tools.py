import random
WEATHER_POSSIBILITIES = ['sunny', 'rainy', 'cloudy', 'windy']

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # 50% chance to always return sunny
    if random.random() < 0.5:
        weather = "sunny"
        message = f"It's always sunny in {city}!"
    else:
        weather = random.choice(WEATHER_POSSIBILITIES)
        message = f"It's {weather} in {city}!"
    return message


def make_answer_about_job_description(llm):
    """Factory that returns a tool bound to the given LLM."""

    def answer_about_job_description(question: str, info_abt_job: str) -> str:
        """Based on the job description, call the llm and answer questions."""
        prompt = f"Based on the following job description: {info_abt_job}, answer the question: {question}"
        return llm.invoke(prompt).content

    return answer_about_job_description
